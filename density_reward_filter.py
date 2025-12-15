import os
os.environ["RAY_DEDUP_LOGS"] = "0"

from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import walker_delta_args, random_circular_orbit , orbitalMotion
from bsk_rl import SatelliteTasking
from bsk_rl.utils.orbital import orbitalMotion, rv2HN
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
import ray
from ray import tune
from ray.tune.registry import register_env
from Basilisk.simulation import msisAtmosphere, dragDynamicEffector, exponentialAtmosphere,facetDragDynamicEffector
from Basilisk.architecture import messaging
from Basilisk.utilities import macros
import gymnasium as gym
from typing import Callable, Optional
import random
from bsk_rl.utils.functional import collect_default_args, default_args
import time
from bsk_rl.sim.world import BasicWorldModel, GroundStationWorldModel
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
from bsk_rl.data.base import GlobalReward, DataStore, Data
from collections import deque

# ✅ 定义正确单位的引力常数
MU_EARTH = orbitalMotion.MU_EARTH * 1e9  # km³/s² → m³/s²

# Instead, use Basilisk's native MU directly
MU_EARTH_KM = orbitalMotion.MU_EARTH  # Already in km³/s²

# Define a shared log file path
LOG_FILE = "/workspace/learn_basilisk/logs/density_altitude_log.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Initialize the log file with headers if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "worker_pid",
            "step",
            "density",
            "alt_km",
            "alt_deviation_km",
            "semi_axis_km",
            "semi_axis_km_filtered",
            "rv2a_deviation_km",
            "rv2a_deviation_km_filtered",
            "sim_time",
            "reward",
            "dv_available",
            "eccentricity",
            "true_anomaly_deg",
        ])

# ==========================================
# 轨道力学辅助函数（使用 Basilisk 内置方法）
# ==========================================

def _calculate_sma_basilisk(r: np.ndarray, v: np.ndarray) -> float:
    """使用 Basilisk 官方方法计算半长轴
    
    Args:
        r: 位置向量 [m]
        v: 速度向量 [m/s]
    
    Returns:
        半长轴 [km]  # ✅ 修正：Basilisk 返回的是 km！
    """
    try:
        # ✅ 使用 Basilisk 原生的 MU（单位是 km³/s²）
        mu = orbitalMotion.MU_EARTH  # 不要乘 1e9！
        
        # ✅ 输入转换为 km 和 km/s
        r_km = np.array(r, dtype=float) / 1000.0
        v_km_s = np.array(v, dtype=float) / 1000.0
        
        oe = orbitalMotion.rv2elem(mu=mu, rVec=r_km, vVec=v_km_s)
        return float(oe.a)  # 返回 km
    except Exception as e:
        print(f"[_calculate_sma_basilisk] Error: {e}")
        # 备用方法：能量法（使用正确单位）
        r_norm = np.linalg.norm(r)  # m
        v_norm = np.linalg.norm(v)  # m/s
        mu_m = orbitalMotion.MU_EARTH * 1e9  # km³/s² → m³/s²
        specific_energy = (v_norm**2) / 2.0 - mu_m / r_norm
        if abs(specific_energy) > 1e-10:
            return float(-mu_m / (2.0 * specific_energy)) / 1000.0  # m → km
        else:
            return float('inf')


def _true_anomaly(r: np.ndarray, v: np.ndarray) -> Optional[float]:
    """计算真近点角 [rad] - 使用 Basilisk
    
    Args:
        r: 位置向量 [m]
        v: 速度向量 [m/s]
    
    Returns:
        真近点角，范围 [0, 2π]；失败返回 None
    """
    try:
        # ✅ 修复：和 get_true_anomaly_deg() 使用相同的逻辑
        r_km = np.array(r, dtype=float) / 1000.0
        v_km_s = np.array(v, dtype=float) / 1000.0
        oe = orbitalMotion.rv2elem(mu=MU_EARTH_KM, rVec=r_km, vVec=v_km_s)
        return float(oe.f)  # 返回弧度
    except Exception as e:
        print(f"[_true_anomaly] Error: {e}")
        return None


def _is_at_apse(r: np.ndarray, v: np.ndarray, 
                apse_type: str = "both",
                angle_tol_deg: float = 5.0) -> bool:
    """判断是否在近地点/远地点附近
    
    Args:
        r: 位置向量 [m]
        v: 速度向量 [m/s]
        apse_type: "periapsis"(近地点), "apoapsis"(远地点), "both"(任意拱点)
        angle_tol_deg: 角度容差 [度]
    
    Returns:
        是否在指定拱点附近
    """
    nu = _true_anomaly(r, v)
    if nu is None:
        return True  # 无法判定时默认允许
    
    angle_tol = np.deg2rad(angle_tol_deg)
    
    if apse_type == "periapsis":
        # 近地点: nu ≈ 0 或 2π
        angle_diff = min(abs(nu), abs(nu - 2*np.pi))
        return angle_diff <= angle_tol
    
    elif apse_type == "apoapsis":
        # 远地点: nu ≈ π
        angle_diff = abs(nu - np.pi)
        return angle_diff <= angle_tol
    
    elif apse_type == "both":
        # 任意拱点
        periapsis_diff = min(abs(nu), abs(nu - 2*np.pi))
        apoapsis_diff = abs(nu - np.pi)
        return min(periapsis_diff, apoapsis_diff) <= angle_tol
    
    else:
        return True

# ==========================================
# 滤波器类定义
# ==========================================

class SMAFilter:
    """轨道半长轴滤波器
    
    支持多种滤波方法：
    - 移动平均滤波 (moving_average)
    - 指数移动平均滤波 (exponential_moving_average / EMA)
    - 一阶低通滤波 (low_pass)
    """
    
    def __init__(self, method: str = "ema", window_size: int = 10, alpha: float = 0.1):
        """
        初始化滤波器
        
        Args:
            method: 滤波方法 ("moving_average", "ema", "low_pass")
            window_size: 移动平均窗口大小
            alpha: EMA/低通滤波的平滑系数 (0 < alpha <= 1)，越小越平滑
        """
        self.method = method
        self.window_size = window_size
        self.alpha = alpha
        
        # 移动平均缓存
        self._buffer = deque(maxlen=window_size)
        
        # EMA/低通滤波状态
        self._filtered_value = None
        self._initialized = False
    
    def reset(self):
        """重置滤波器状态"""
        self._buffer.clear()
        self._filtered_value = None
        self._initialized = False
    
    def update(self, raw_value: float) -> float:
        """
        更新滤波器并返回滤波后的值
        
        Args:
            raw_value: 原始测量值
            
        Returns:
            滤波后的值
        """
        if self.method == "moving_average":
            return self._moving_average(raw_value)
        elif self.method == "ema":
            return self._exponential_moving_average(raw_value)
        elif self.method == "low_pass":
            return self._low_pass_filter(raw_value)
        else:
            return raw_value  # 不滤波
    
    def _moving_average(self, raw_value: float) -> float:
        """移动平均滤波"""
        self._buffer.append(raw_value)
        return float(np.mean(self._buffer))
    
    def _exponential_moving_average(self, raw_value: float) -> float:
        """指数移动平均滤波 (EMA)"""
        if not self._initialized:
            self._filtered_value = raw_value
            self._initialized = True
        else:
            # EMA 公式: filtered = alpha * raw + (1 - alpha) * filtered_prev
            self._filtered_value = self.alpha * raw_value + (1 - self.alpha) * self._filtered_value
        return float(self._filtered_value)
    
    def _low_pass_filter(self, raw_value: float) -> float:
        """一阶低通滤波（与 EMA 类似，但物理意义更明确）"""
        return self._exponential_moving_average(raw_value)
    
    @property
    def current_value(self) -> Optional[float]:
        """获取当前滤波值（不更新）"""
        return self._filtered_value


# 全局滤波器实例（每个卫星一个）
_sma_filters = {}

def get_sma_filter(sat_id: str = "default") -> SMAFilter:
    """获取或创建指定卫星的滤波器"""
    if sat_id not in _sma_filters:
        _sma_filters[sat_id] = SMAFilter(method="moving_average", window_size=150)
    return _sma_filters[sat_id]

def reset_sma_filter(sat_id: str = "default"):
    """重置指定卫星的滤波器"""
    if sat_id in _sma_filters:
        _sma_filters[sat_id].reset()


#---------------自定义rewarder-----------------

# ==========================================
# 1. 定义数据载体 (Data Carrier)
# ==========================================
class StationKeepingData(Data):
    def __init__(self, position=None, velocity=None, fuel_mass=0.0, sma_filtered=None):
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0])
        self.velocity = velocity if velocity is not None else np.array([0.0, 0.0, 0.0])
        self.fuel_mass = fuel_mass
        self.sma_filtered = sma_filtered
        
    def __add__(self, other):
        return StationKeepingData(other.position, other.velocity, other.fuel_mass, other.sma_filtered)


# 2. 定义数据存储器 (Data Store)
class StationKeepingDataStore(DataStore):
    data_type = StationKeepingData 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sma_filter = SMAFilter(method="moving_average", window_size=150)
    
    def get_log_state(self):
        r_BN_N = np.array(self.satellite.dynamics.r_BN_N)
        v_BN_N = np.array(self.satellite.dynamics.v_BN_N)
        fuel = self.satellite.fsw.dv_available
        
        # ✅ 计算半长轴 [km] 并滤波
        sma_km = _calculate_sma_basilisk(r_BN_N, v_BN_N)
        sma_filtered_km = self._sma_filter.update(sma_km)

        return r_BN_N, v_BN_N, fuel, sma_filtered_km

    def compare_log_states(self, old_state, new_state):
        return StationKeepingData(
            position=new_state[0], 
            velocity=new_state[1], 
            fuel_mass=new_state[2],
            sma_filtered=new_state[3]
        )


# 3. 定义Rewarder
class StationKeepingReward(GlobalReward):
    data_store_type = StationKeepingDataStore

    def __init__(self, target_alt_km=293.0, dist_penalty_scale=0.1, fuel_cost=0.1, **kwargs):
        super().__init__()
        # ✅ 目标半长轴 [km]（不需要乘 1000 了）
        self.target_sma_km = orbitalMotion.REQ_EARTH + target_alt_km
        self.dist_penalty_scale = dist_penalty_scale
        self.fuel_cost = fuel_cost
        self.last_fuel = {} 

    def calculate_reward(self, new_data_dict):
        rewards = {}
        
        for sat_id, sat_data in new_data_dict.items():
            r_BN_N = sat_data.position
            v_BN_N = sat_data.velocity
            curr_fuel = sat_data.fuel_mass
            
            # ✅ 滤波后的半长轴已经是 km
            if sat_data.sma_filtered is not None:
                sma_km = sat_data.sma_filtered
            else:
                sma_km = _calculate_sma_basilisk(r_BN_N, v_BN_N)
            
            # ✅ 直接计算误差（都是 km）
            error_km = sma_km - self.target_sma_km  # 注意：保留符号，不取绝对值
            
            if abs(error_km) > 0.2:
                sma_error_km = (abs(error_km) * 5)**2
                r_dist = -sma_error_km * self.dist_penalty_scale
            else:
                r_dist = 1
            
            # ✅ 先计算燃料消耗（必须在使用之前定义）
            r_fuel = 0.0
            fuel_consumed = 0.0  # 初始化默认值
            if sat_id in self.last_fuel:
                fuel_consumed = self.last_fuel[sat_id] - curr_fuel
                if fuel_consumed > 1e-6:
                    r_fuel = -fuel_consumed * self.fuel_cost
            
            # ✅ 轨道机动奖励（现在可以安全使用 fuel_consumed）
            r_orbital_maneuver = 0.0
            if fuel_consumed > 1e-6:  # 确保有推力发生
                nu = _true_anomaly(r_BN_N, v_BN_N)
                
                if nu is not None:  # 确保真近点角计算成功
                    # 判断升轨还是降轨需求
                    if error_km < 0:  # 需要升轨（当前轨道太低）
                        # 在远地点推力最优（nu ≈ 180°）
                        if abs(nu - np.pi) < np.deg2rad(5):
                            r_orbital_maneuver = 1.0  # 奖励
                        else:
                            r_orbital_maneuver = -0.5  # 惩罚
                    
                    elif error_km > 0:  # 需要降轨（当前轨道太高）
                        # 在近地点推力最优（nu ≈ 0°）
                        periapsis_diff = min(abs(nu), abs(nu - 2*np.pi))
                        if periapsis_diff < np.deg2rad(5):
                            r_orbital_maneuver = 1.0
                        else:
                            r_orbital_maneuver = -0.5
            
            # 在接近目标轨道时，惩罚过度推力
            if abs(error_km) < 1.0:  # 接近目标
                if fuel_consumed > 1e-6:
                    r_unnecessary_thrust = -2.0  # 额外惩罚
                else:
                    r_unnecessary_thrust = 0.0
            else:
                r_unnecessary_thrust = 0.0
            
            # 更新燃料记录
            self.last_fuel[sat_id] = curr_fuel 
            
            # 总奖励
            rewards[sat_id] = r_dist + r_fuel + r_orbital_maneuver + r_unnecessary_thrust

        return rewards

#--------------------------------------------------

class ExChangeWorld(GroundStationWorldModel):

    @default_args(
        planetRadius=orbitalMotion.REQ_EARTH * 1e3,
        baseDensity=1.5,
        scaleHeight=11000,
    )
    def setup_atmosphere_density_model(
        self,
        planetRadius: float,
        baseDensity: float,
        scaleHeight: float,
        priority: int = 1000,
        **kwargs,
    ) -> None:
        print("baseDensity:", baseDensity)
        self.densityModel = exponentialAtmosphere.ExponentialAtmosphere()
        self.densityModel.ModelTag = "expDensity"
        self.densityModel.planetRadius = orbitalMotion.REQ_EARTH * 1000.0
        self.densityModel.baseDensity = baseDensity     
        self.densityModel.scaleHeight = scaleHeight
        self.densityModel.planetPosInMsg.subscribeTo(
            self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
        )
        self.simulator.AddModelToTask(
            self.world_task_name, self.densityModel, ModelPriority=1000
        )


class ExponentialDynModel(dyn.FullFeaturedDynModel):

    @classmethod
    def _requires_world(cls):
        return [ExChangeWorld]

    def setup_density_model(self) -> None:
        self.world.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsg)

    @default_args(dragCoeff=2.2)
    def setup_drag_effector(self,
        width: float = 1.0,
        depth: float = 1.0,
        height: float = 1.0,
        panelArea: float = 10.0,
        dragCoeff: float = 2.2,
        priority: int = 999,
        **kwargs,) -> None:

        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        self.dragEffector.addFacet(
            width * depth, dragCoeff, [1, 0, 0], [height / 2, 0.0, 0]
        )
        self.dragEffector.addFacet(
            width * depth, dragCoeff, [-1, 0, 0], [height / 2, 0.0, 0]
        )
        self.dragEffector.addFacet(
            height * width, dragCoeff, [0, 1, 0], [0, depth / 2, 0]
        )
        self.dragEffector.addFacet(
            height * width, dragCoeff, [0, -1, 0], [0, -depth / 2, 0]
        )
        self.dragEffector.addFacet(
            height * depth, dragCoeff, [0, 0, 1], [0, 0, width / 2]
        )
        self.dragEffector.addFacet(
            height * depth, dragCoeff, [0, 0, -1], [0, 0, -width / 2]
        )
        self.dragEffector.addFacet(
            panelArea / 2,
            dragCoeff,
            [0, 1, 0],
            [0, height, 0],
        )
        self.dragEffector.addFacet(
            panelArea / 2,
            dragCoeff,
            [0, -1, 0],
            [0, height, 0],
        )
        self.dragEffector.atmoDensInMsg.subscribeTo(
            self.world.densityModel.envOutMsgs[-1]
        )
        self.scObject.addDynamicEffector(self.dragEffector)

        self.simulator.AddModelToTask(
            self.task_name, self.dragEffector, ModelPriority=priority
        )


def altitude_deviation(sat) -> float:
    r_BN_N = np.array(sat.dynamics.r_BN_N)
    current_alt = np.linalg.norm(r_BN_N) / 1000.0 - orbitalMotion.REQ_EARTH
    reference_alt = 293.0
    deviation = current_alt - reference_alt
    return deviation

def current_altitude(sat) -> float:
    r_BN_N = np.array(sat.dynamics.r_BN_N)
    alt = np.linalg.norm(r_BN_N) / 1000.0 - orbitalMotion.REQ_EARTH
    return alt


def rv2a_deviation(sat) -> np.ndarray:
    """计算滤波后的轨道半长轴偏移量 [km]"""
    r = np.array(sat.dynamics.r_BN_N)
    v = np.array(sat.dynamics.v_BN_N)
    
    # ✅ 现在 _calculate_sma_basilisk 直接返回 km
    current_a_km = _calculate_sma_basilisk(r, v)
    
    sat_filter = get_sma_filter(sat.name)
    current_a_filtered_km = sat_filter.update(current_a_km)
    
    reference_a_km = orbitalMotion.REQ_EARTH + 293.0
    deviation = current_a_filtered_km - reference_a_km
    return np.array([deviation])

def rv2a_deviation_raw(sat) -> np.ndarray:
    """计算原始（未滤波）的轨道半长轴偏移量 [km]"""
    r = np.array(sat.dynamics.r_BN_N)
    v = np.array(sat.dynamics.v_BN_N)
    
    # ✅ 直接返回 km
    current_a_km = _calculate_sma_basilisk(r, v)
    reference_a_km = orbitalMotion.REQ_EARTH + 293.0
    deviation = current_a_km - reference_a_km
    return np.array([deviation])

def fuel_remaining(sat) -> np.ndarray:
    try:
        dv = sat.fsw.dv_available
    except Exception:
        dv = 200.0
    return np.array([dv])
    
def get_current_density(sat) -> np.ndarray:
    density = sat.dynamics.world.densityModel.baseDensity
    return np.array([density])


class ImagingMagicOrbitManeuverFSWModel(fsw.MagicOrbitalManeuverFSWModel, fsw.ImagingFSWModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


def get_thrust_gate_status(sat) -> np.ndarray:
    """返回当前是否允许推力 [0 或 1]"""
    try:
        r = np.array(sat.dynamics.r_BN_N)
        v = np.array(sat.dynamics.v_BN_N)
        can_thrust = _is_at_apse(r, v, apse_type="both", angle_tol_deg=5.0)
        return np.array([1.0 if can_thrust else 0.0])
    except Exception:
        return np.array([1.0])

def get_eccentricity(sat) -> np.ndarray:
    """返回轨道偏心率"""
    try:
        r = np.array(sat.dynamics.r_BN_N)  # meters
        v = np.array(sat.dynamics.v_BN_N)  # m/s
        # Convert to km for rv2elem
        r_km = r / 1000.0
        v_km_s = v / 1000.0
        oe = orbitalMotion.rv2elem(mu=MU_EARTH_KM, rVec=r_km, vVec=v_km_s)
        return np.array([oe.e])
    except Exception:
        return np.array([0.0])

def get_true_anomaly_deg(sat) -> np.ndarray:
    """返回真近点角 [度]"""
    try:
        r = np.array(sat.dynamics.r_BN_N)  # meters
        v = np.array(sat.dynamics.v_BN_N)  # m/s
        # Convert to km for rv2elem
        r_km = r / 1000.0
        v_km_s = v / 1000.0
        oe = orbitalMotion.rv2elem(mu=MU_EARTH_KM, rVec=r_km, vVec=v_km_s)
        return np.array([np.rad2deg(oe.f)])
    except Exception:
        return np.array([0.0])

class MySatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="rv2a_deviation", fn=rv2a_deviation, norm=1.0),
            dict(prop="fuel_remaining", fn=fuel_remaining, norm=200.0),
            dict(prop="current_density", fn=get_current_density, norm=1.0),
            dict(prop="thrust_gate_status", fn=get_thrust_gate_status, norm=1.0),
            dict(prop="eccentricity", fn=get_eccentricity, norm=1),
            dict(prop="true_anomaly", fn=get_true_anomaly_deg, norm=180.0),
        ),
    ]   
    action_spec = [
        act.ImpulsiveThrust(
            name="DragThrust",
            max_dv=0.5,
            max_drift_duration=300.0,
            fsw_action= None,
        ),
    ] 
    dyn_type = ExponentialDynModel 
    fsw_type = fsw.MagicOrbitalManeuverFSWModel

    @default_args(dv_available_init=293.0)
    def setup_fuel(self, dv_available_init: float = 293.0, **kwargs) -> None:
        super().setup_fuel(dv_available_init=dv_available_init, **kwargs)


class DensityWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        density_schedule: Optional[Callable[[float], float]] = None,
        thrust_gate_type: str = "none",
        thrust_gate_angle_tol: float = 5.0,
    ):
        super().__init__(env)
        self.density_schedule = density_schedule
        self.step_count  = 0
        self._density_model = None
        self.pid = os.getpid()
        
        self.thrust_gate_type = thrust_gate_type
        self.thrust_gate_angle_tol = thrust_gate_angle_tol
        
        self.thrust_blocked_count = 0
        self.thrust_allowed_count = 0
        
        self._sma_filter = SMAFilter(method="moving_average", window_size=150)
    
    @property
    def density_model(self):
        if self._density_model is None:
            self._density_model = self.unwrapped.satellite.dynamics.world.densityModel
        return self._density_model
    
    def _get_sim_time(self) -> float:
        return getattr(self.unwrapped.simulator, "sim_time", 0.0)
    
    def _log_state(self, reward=0.0):
        """Helper to log current state to CSV"""
        try:
            r_BN_N = np.array(self.unwrapped.satellite.dynamics.r_BN_N)
            v_BN_N = np.array(self.unwrapped.satellite.dynamics.v_BN_N)
            alt_km = np.linalg.norm(r_BN_N) / 1000.0 - orbitalMotion.REQ_EARTH
            deviation_km = alt_km - 293.0
            
            # ✅ 现在 _calculate_sma_basilisk 返回 km
            semi_axis_km = _calculate_sma_basilisk(r_BN_N, v_BN_N)
            semi_axis_km_filtered = self._sma_filter.update(semi_axis_km)
            
            sim_time = self._get_sim_time()

            try:
                ref_sma_km = orbitalMotion.REQ_EARTH + 293.0
                rv2a_deviation_km = semi_axis_km - ref_sma_km
                rv2a_deviation_km_filtered = semi_axis_km_filtered - ref_sma_km
            except Exception:
                rv2a_deviation_km = float("nan")
                rv2a_deviation_km_filtered = float("nan")

            try:
                dv_available = self.unwrapped.satellite.fsw.dv_available
            except Exception:
                dv_available = float("nan")

            # ✅ 计算轨道六根数
            try:
                r_km = r_BN_N / 1000.0
                v_km_s = v_BN_N / 1000.0
                oe = orbitalMotion.rv2elem(mu=MU_EARTH_KM, rVec=r_km, vVec=v_km_s)
                eccentricity = float(oe.e)
                true_anomaly_deg = float(np.rad2deg(oe.f))
            except Exception as e:
                print(f"[_log_state] rv2elem error: {e}")
                eccentricity = float("nan")
                true_anomaly_deg = float("nan")

            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.pid,
                    self.step_count,
                    self.density_model.baseDensity,
                    alt_km,
                    deviation_km,
                    semi_axis_km,
                    semi_axis_km_filtered,  
                    rv2a_deviation_km,
                    rv2a_deviation_km_filtered,  
                    sim_time,
                    reward,
                    dv_available,
                    eccentricity,
                    true_anomaly_deg,
                ])
        except Exception as e:
            print(f"[DensityWrapper] Logging error: {e}")
            import traceback
            traceback.print_exc()

    def reset(self, **kwargs):
        self.step_count = 0
        self._density_model = None
        
        self.thrust_blocked_count = 0
        self.thrust_allowed_count = 0
        
        self._sma_filter.reset()
        
        sat_name = getattr(self.unwrapped, 'satellite', None)
        if sat_name and hasattr(sat_name, 'name'):
            reset_sma_filter(sat_name.name)
        
        obs, info = self.env.reset(**kwargs)
        self._update_density()
        self._log_state(reward=0.0)
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        
        action_array = np.array(action, dtype=np.float32, copy=True)
        
        if self.thrust_gate_type != "none":
            allow_thrust = self._check_thrust_gate()
            
            if not allow_thrust:
                action_array[:3] = 0.0
                self.thrust_blocked_count += 1
            else:
                self.thrust_allowed_count += 1
        
        obs, reward, terminated, truncated, info = self.env.step(action_array)
        self._update_density()
        
        if isinstance(reward, dict):
            reward_value = sum(reward.values()) / len(reward) if reward else 0.0
        else:
            reward_value = float(reward)
        
        self._log_state(reward=reward_value)
        
        info = dict(info) if info is not None else {}
        if self.thrust_gate_type != "none":
            info["thrust_gated"] = not allow_thrust
            info["thrust_gate_stats"] = {
                "blocked": self.thrust_blocked_count,
                "allowed": self.thrust_allowed_count,
            }
        
        return obs, reward, terminated, truncated, info
    
    def _check_thrust_gate(self) -> bool:
        """检查当前是否允许推力"""
        try:
            r_BN_N = np.array(self.unwrapped.satellite.dynamics.r_BN_N)
            v_BN_N = np.array(self.unwrapped.satellite.dynamics.v_BN_N)
            return _is_at_apse(
                r_BN_N, v_BN_N,
                apse_type=self.thrust_gate_type,
                angle_tol_deg=self.thrust_gate_angle_tol
            )
        except Exception as e:
            print(f"[DensityWrapper] Thrust gate check failed: {e}")
            return True

    def _update_density(self):
        if self.density_schedule is not None:
            sim_time = self._get_sim_time()
            new_density = self.density_schedule(sim_time)
            self.density_model.baseDensity = new_density
     
    def set_density(self, density: float):
        self.density_model.baseDensity = density


def step_density(step: int) -> float:
    return 0.0

def time_density(sim_time: float) -> float:
    orbit_period = 5700.0
    
    if sim_time < 1 * orbit_period:
        return 0.3
    elif sim_time < 10 * orbit_period:
        return 1.3
    elif sim_time < 20 * orbit_period:
        return 0.3
    else:
        return 0.0


my_rewarder = StationKeepingReward(target_alt_km=293.0, dist_penalty_scale=0.5, fuel_cost=3)

oe = random_circular_orbit(alt=300.0, i=0.0, Omega=0.0, f=0.0)
sat_args = {"oe": oe}

gs_data = [
    dict(name="GS_Alaska",   lat=64.0,  long=-147.5, elev=0.0),
    dict(name="GS_Norway",   lat=69.0,  long=  18.9, elev=0.0),
    dict(name="GS_Australia",lat=-35.3, long= 149.1, elev=0.0),
]

sat = MySatellite(name="EO1", sat_args=sat_args)

env_args = dict(
    satellite=sat,
    rewarder=my_rewarder,
    failure_penalty=-100.0,
    time_limit=5700.0 * 30,
    log_level="WARNING",
    world_args=dict(
        groundStationsData=gs_data,
        gsMinimumElevation=np.radians(10.0),
        gsMaximumRange=-1,
        utc_init="2018 SEP 29 21:00:00.000 (UTC)",
    ),   
)


def make_wrapped_env(env_config):
    config = env_config.copy()
    density_schedule = config.pop("density_schedule", None)
    episode_callback = config.pop("episode_data_callback", None)
    
    thrust_gate_type = config.pop("thrust_gate_type", "none")
    thrust_gate_angle_tol = config.pop("thrust_gate_angle_tol", 5.0)
    
    base_env = SatelliteTasking(**config)
    
    if density_schedule is not None or thrust_gate_type != "none":
        wrapper_env = DensityWrapper(
            base_env,
            density_schedule=density_schedule,
            thrust_gate_type=thrust_gate_type,
            thrust_gate_angle_tol=thrust_gate_angle_tol
        )
        return wrapper_env
    return base_env


register_env("SatelliteTasking-Density", make_wrapped_env)


def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (95 * 60)

    data = dict(
        reward=reward,
        alive=float(env.satellite.is_alive()),
        rw_status_valid=float(env.satellite.dynamics.rw_speeds_valid()),
        battery_status_valid=float(env.satellite.dynamics.battery_valid()),
        orbits_complete=orbits,
    )
    if orbits > 0:
        data["reward_per_orbit"] = reward / orbits
    if not env.satellite.is_alive():
        data["orbits_complete_partial_only"] = orbits

    return data


N_CPUS = 8
N_GPUS = 1
training_args = dict(
    lr=0.00001,
    gamma=0.999,
    train_batch_size=1000 * (N_CPUS-1),
    num_sgd_iter=20,
    model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False),
    lambda_=0.95,
    use_kl_loss=True,
    clip_param=0.1,
    grad_clip=0.5,
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(num_env_runners=N_CPUS-1, sample_timeout_s=1000.0)
    .environment(
        env="SatelliteTasking-Density",
        env_config=dict(**env_args,
        episode_data_callback=episode_data_callback,
        density_schedule=time_density,
        thrust_gate_type="both",
        thrust_gate_angle_tol=5.0,
        ),
    )
    .reporting(
        metrics_num_episodes_for_smoothing=1,
        metrics_episode_collection_timeout_s=180,
    )
    .checkpointing(export_native_model_files=True)
    .framework(framework="torch")
    .debugging(log_level="INFO")
    .callbacks(WrappedEpisodeDataCallbacks)
)   


ray.init(
    ignore_reinit_error=True,
    num_cpus=N_CPUS,
    object_store_memory=8_000_000_000,
)

try:
    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 2000},
        checkpoint_freq=50,
        checkpoint_at_end=True,
        storage_path="/workspace/learn_basilisk/ray_results",
        verbose=1,
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
finally:
    ray.shutdown()


# --- Plotting Section ---
# print("Generating plots from", LOG_FILE)
# try:
#     data_points = []
#     with open(LOG_FILE, "r") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             data_points.append(row)
    
#     if data_points:
#         import pandas as pd
#         df = pd.DataFrame(data_points)
#         cols = [
#             "step",
#             "density",
#             "alt_km",
#             "alt_deviation_km",
#             "semi_axis_km",
#             "semi_axis_km_filtered",
#             "rv2a_deviation_km",
#             "rv2a_deviation_km_filtered",
#             "sim_time",
#             "reward",
#             "dv_available",
#             "eccentricity",
#             "true_anomaly_deg",
#         ]
#         for c in cols:
#             if c in df.columns:
#                 df[c] = pd.to_numeric(df[c], errors='coerce')
        
#         worker_ids = df["worker_pid"].unique()
#         for pid in worker_ids:
#             g = df[df["worker_pid"] == pid].sort_values("sim_time")

#             fig, axes = plt.subplots(6, 1, figsize=(20, 24), sharex=True)
            
#             # Density
#             axes[0].plot(g["sim_time"], g["density"], alpha=0.8)
#             axes[0].set_ylabel("Density (kg/m^3)")
#             axes[0].set_title(f"Worker {pid} - Atmospheric Density over Simulation Time")
#             axes[0].grid(True)
            
#             # Altitude
#             axes[1].plot(g["sim_time"], g["alt_km"], alpha=0.8)
#             axes[1].axhline(y=293.0, color='green', linestyle='--', label='Target 293 km')
#             axes[1].set_ylabel("Altitude (km)")
#             axes[1].set_title(f"Worker {pid} - Satellite Altitude over Simulation Time")
#             axes[1].legend()
#             axes[1].grid(True)

#             # Semi-Major Axis: Raw vs Filtered
#             if "semi_axis_km" in g.columns and "semi_axis_km_filtered" in g.columns:
#                 axes[2].plot(g["sim_time"], g["semi_axis_km"], alpha=0.4, label="Raw SMA", color='blue')
#                 axes[2].plot(g["sim_time"], g["semi_axis_km_filtered"], alpha=0.9, label="Filtered SMA", color='red', linewidth=2)
#                 target_sma = orbitalMotion.REQ_EARTH + 293.0
#                 axes[2].axhline(y=target_sma, color='green', linestyle='--', label=f'Target {target_sma:.1f} km')
#                 axes[2].set_ylabel("Semi-Major Axis (km)")
#                 axes[2].set_title(f"Worker {pid} - Semi-Major Axis (Raw vs Filtered)")
#                 axes[2].legend()
#                 axes[2].grid(True)

#             # SMA Deviation: Raw vs Filtered
#             if "rv2a_deviation_km" in g.columns and "rv2a_deviation_km_filtered" in g.columns:
#                 axes[3].plot(g["sim_time"], g["rv2a_deviation_km"], alpha=0.4, label="Raw Deviation", color='blue')
#                 axes[3].plot(g["sim_time"], g["rv2a_deviation_km_filtered"], alpha=0.9, label="Filtered Deviation", color='red', linewidth=2)
#                 axes[3].axhline(y=0, color='green', linestyle='--', label='Target')
#                 axes[3].set_ylabel("SMA Deviation (km)")
#                 axes[3].set_title(f"Worker {pid} - SMA Deviation from Target (Raw vs Filtered)")
#                 axes[3].legend()
#                 axes[3].grid(True)

#             # Eccentricity and True Anomaly
#             if "eccentricity" in g.columns:
#                 ax4 = axes[4]
#                 ax4.plot(g["sim_time"], g["eccentricity"], alpha=0.8, color='purple')
#                 ax4.set_ylabel("Eccentricity", color='purple')
#                 ax4.set_title(f"Worker {pid} - Eccentricity and True Anomaly")
#                 ax4.tick_params(axis='y', labelcolor='purple')
#                 ax4.grid(True)
                
#                 if "true_anomaly_deg" in g.columns:
#                     ax4_twin = ax4.twinx()
#                     ax4_twin.plot(g["sim_time"], g["true_anomaly_deg"], alpha=0.8, color='orange')
#                     ax4_twin.set_ylabel("True Anomaly (deg)", color='orange')
#                     ax4_twin.tick_params(axis='y', labelcolor='orange')

#             # Reward
#             if "reward" in g.columns:
#                 axes[5].plot(g["sim_time"], g["reward"], alpha=0.8)
#                 axes[5].set_ylabel("Reward")
#                 axes[5].set_xlabel("Simulation Time (s)")
#                 axes[5].set_title(f"Worker {pid} - Reward over Simulation Time")
#                 axes[5].grid(True)
            
#             out_png = f"/workspace/learn_basilisk/logs/worker_{pid}_timeseries.png"
#             plt.tight_layout()
#             plt.savefig(out_png)
#             plt.close(fig)
#             print(f"Plot saved to {out_png}")
            
#     else:
#         print("No data points found for plotting.")
# except Exception as e:
#     print("Error during plotting:", e)
#     import traceback
#     traceback.print_exc()



