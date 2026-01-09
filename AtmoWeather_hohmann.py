import os
os.environ["RAY_DEDUP_LOGS"] = "0"

from bsk_rl import sats, act, obs
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import random_circular_orbit, orbitalMotion
from bsk_rl import SatelliteTasking
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import ray
from ray import tune
from ray.tune.registry import register_env
from Basilisk.simulation import exponentialAtmosphere, facetDragDynamicEffector , msisAtmosphere
import gymnasium as gym
from typing import Callable, Optional
from bsk_rl.utils.functional import default_args
from bsk_rl.sim.world import GroundStationWorldModel
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bsk_rl.data.base import GlobalReward, DataStore, Data
from collections import deque
from Basilisk.utilities import simIncludeGravBody

# always import the Basilisk messaging support
from Basilisk.architecture import messaging
from bsk_rl.act.continuous_actions import ContinuousAction
from gymnasium import spaces

import torch 
import torch.nn as nn 
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2 
from ray.rllib.models import ModelCatalog
# ==========================================
# 全局配置参数
# ==========================================
Target_sma = 300.0  # 目标轨道高度 [km]
Max_Hohmann_dh = 1000.0  # 最大一次提升高度 m
Max_fuel = 500      # 最大燃料量 [m/s]
# Basilisk 原生引力常数 [km³/s²]
MU_EARTH = orbitalMotion.MU_EARTH
# REQ_EARTH = orbitalMotion.REQ_EARTH
REQ_EARTH = 6371  # 使用6371km简化，因为bsk-rl创建轨道时，采用的参数是6371km
V_init = np.sqrt(MU_EARTH / (REQ_EARTH + Target_sma))  # 初始轨道速度 [km/s]
LOG_FILE = "/workspace/learn_basilisk/RL-Hohmann-True/Model/With_Nothing/logs/density_altitude_log.csv"
RAY_LOG_FILE = "/workspace/learn_basilisk/RL-Hohmann-True/Model/With_Nothing/ray_log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
PLOT_RESULTS = False
# 初始化日志文件
if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "worker_pid", "step", "density", "alt_km", "v_km_s", "alt_deviation_km",
                "semi_axis_km", "rv2a_deviation_km",
                "sim_time", "reward", "dv_available",
                "ap", "f107", "atmospheric_density",  # ✅ 新增：大气密度列
            ])
# ==========================================
# 轨道力学辅助函数
# ==========================================

# filepath: /workspace/learn_basilisk/atmoweather_density/atmoweather_change.py
def _calculate_sma(r: np.ndarray, v: np.ndarray) -> tuple:
    """计算半长轴
    
    返回: (sma_km, is_valid) - 半长轴值和有效性标志
    """
    try:
        r_km = np.array(r, dtype=float) / 1000.0
        v_km_s = np.array(v, dtype=float) / 1000.0
        oe = orbitalMotion.rv2elem(mu=MU_EARTH, rVec=r_km, vVec=v_km_s)
        # rv2elem 可能在遇到 NaN 时返回 None（见包源码），因此需要检查
        if oe is None:
            print("[WARN] _calculate_sma: rv2elem returned None, using default SMA")
            return (REQ_EARTH + Target_sma, False)
        a_val = float(oe.a)
        return (a_val, True)  # 返回是km
    except Exception as e:
        print(f"[WARN] _calculate_sma: Exception {e}, using default SMA")
        return (REQ_EARTH + Target_sma, False)

# ==========================================
# 自定义 Rewarder
# ==========================================

class StationKeepingData(Data):
    def __init__(self, position=None, velocity=None, fuel_mass=0.0, sim_time: float = 0.0):
        self.position = position if position is not None else np.array([0.0, 0.0, 0.0])
        self.velocity = velocity if velocity is not None else np.array([0.0, 0.0, 0.0])
        self.fuel_mass = fuel_mass
        self.sim_time = float(sim_time)
    def __add__(self, other):
        return StationKeepingData(other.position, other.velocity, other.fuel_mass, other.sim_time)

class StationKeepingDataStore(DataStore):
    data_type = StationKeepingData 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def get_log_state(self):
        r_BN_N = np.array(self.satellite.dynamics.r_BN_N)
        v_BN_N = np.array(self.satellite.dynamics.v_BN_N)
        fuel = self.satellite.fsw.dv_available
        sim_time = float(getattr(self.satellite.simulator, "sim_time", 0.0))
        return r_BN_N, v_BN_N, fuel, sim_time

    def compare_log_states(self, old_state, new_state):
        return StationKeepingData(
            position=new_state[0], 
            velocity=new_state[1], 
            fuel_mass=new_state[2],
            sim_time=new_state[3]
        )

class StationKeepingReward(GlobalReward):
    data_store_type = StationKeepingDataStore
    
    def __init__(self, satellite = None, dist_penalty_scale=0.5, fuel_cost=50, **kwargs):
        super().__init__()
        self.dist_penalty_scale = dist_penalty_scale
        self.satellite  = satellite
        self.fuel_cost = fuel_cost
        self.last_fuel = {} 
        self.last_time = {}
        self.orbit_invalid = False  # 轨道无效标志
        
    def calculate_reward(self, new_data_dict):
        rewards = {}
        self.orbit_invalid = False  # 每次重置

        target_sma_km = self._target_sma_km  # 我只创建了一颗卫星，所以这里可以直接传递进来
        # print(f"target_sma_km: {target_sma_km:.2f}")
        
        for sat_id, sat_data in new_data_dict.items():
            r_BN_N = sat_data.position
            v_BN_N = sat_data.velocity
            # sim_time = sat_data.sim_time
            # alt_km = np.linalg.norm(r_BN_N) / 1000.0 
            sma_km, is_valid = _calculate_sma(r_BN_N, v_BN_N)
            if not is_valid:
                self.orbit_invalid = True
                print(f"[WARN] {sat_id}: Orbit calculation invalid, will terminate episode")
            error_km = sma_km - target_sma_km
            print(f"{sat_id},target_sma_km: {target_sma_km}, sma_km: {sma_km}, error_km: {error_km},abs_error_km={abs(error_km)}")
            curr_fuel = sat_data.fuel_mass

            # if abs(error_km) > 0.5:
            #     r = - 2 * abs(error_km) * self.dist_penalty_scale
            # else:
            #     r = 2 * (1 - abs(error_km)) * self.dist_penalty_scale
            r = 4*(0.5 - abs(error_km))*self.dist_penalty_scale
            print("r = ", r)
            # 燃料消耗惩罚
            r_fuel = 0.0
            fuel_consumed = 0.0
            if sat_id in self.last_fuel:
                fuel_consumed = self.last_fuel[sat_id] - curr_fuel
                r_fuel = - fuel_consumed * self.fuel_cost
            print(f"{sat_id}, fuel_consumed: {fuel_consumed:.2f}, r_fuel: {r_fuel:.2f}")    
            rewards[sat_id] = r  + r_fuel
            self.last_fuel[sat_id] = curr_fuel

        return rewards

# ==========================================
# 读取起始高度数据和目标高度数据
# ==========================================
class AltitudeConfigManager:
    """从 CSV 文件读取初始和目标高度配置"""
    
    def __init__(self, config_file: str):
        self.configs = self._load_configs(config_file)
        self.current_index = 0
    
    def _load_configs(self, config_file: str) -> list:
        """直接读取并返回配置列表，不存在时返回空列表"""
        if not os.path.exists(config_file):
            print(f"[AltitudeConfigManager] Config file not found: {config_file}, using default values")
            return []
        
        try:
            configs = []
            with open(config_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    configs.append({
                        "initial_altitude_km": float(row["initial_km"]),
                        "target_altitude_km": float(row["target_km"]),
                    })
            print(f"[AltitudeConfigManager] Loaded {len(configs)} configurations from {config_file}")
            return configs
        except Exception as e:
            print(f"[AltitudeConfigManager] Error loading config: {e}")
            return []
    
    def get_next_config(self, default_alt: float) -> tuple:
        """获取下一个配置，如果列表为空则使用默认值"""
        if not self.configs:
            return default_alt, default_alt
        
        config = self.configs[self.current_index % len(self.configs)]
        self.current_index += 1
        print(f"[AltitudeConfigManager] Providing config: initial_altitude_km={config['initial_altitude_km']}, target_altitude_km={config['target_altitude_km']}")
        return config["initial_altitude_km"], config["target_altitude_km"]

# ==========================================
# 自定义环境模型
# ==========================================
    # ==========================================
    # 空间天气调度函数
    # ==========================================

# ✅ 全局计数器：使用简单的字典存储（不依赖 multiprocessing）
# 在 Ray Worker 进程中，每个 Worker 会加载一次模块，所以字典会被隔离
_episode_counter_dict = {"counter": 0}

def get_next_episode_date_index():
    """获取下一个 Episode 的日期索引（基于全局计数器）"""
    global _episode_counter_dict
    index = _episode_counter_dict["counter"]
    _episode_counter_dict["counter"] += 1
    return index

class SpaceWeatherProvider:
    """
    从CSV文件读取空间天气参数，支持基于仿真时间的线性插值。
    CSV格式：year, month, day, ap_24_0, ap_3_0, ..., f107_1944_0, f107_24_-24
    
    插值逻辑：
    - 每天零点对应一条数据
    - 仿真时间 t 秒时，计算 t 在当天零点和次日零点之间的比例
    - 对所有参数进行线性插值
    """
    
    SECONDS_PER_DAY = 86400.0  # 一天的秒数
    
    def __init__(self, config_file: str = None, date_config_file: str = None):
        """
        Args:
            config_file: 空间天气数据CSV文件路径
            date_config_file: 日期配置文件路径 (若提供则从此文件读取起始日期)
        """
        self.config_file = config_file
        self.date_config_file = date_config_file
        self.data_by_date = {}  # {(year, month, day): param_dict}
        
        # 定义参数通道顺序（与MSIS端口匹配）
        self.sw_keys = [
            "ap_24_0", "ap_3_0", "ap_3_-3", "ap_3_-6", "ap_3_-9",
            "ap_3_-12", "ap_3_-15", "ap_3_-18", "ap_3_-21", "ap_3_-24",
            "ap_3_-27", "ap_3_-30", "ap_3_-33", "ap_3_-36", "ap_3_-39",
            "ap_3_-42", "ap_3_-45", "ap_3_-48", "ap_3_-51", "ap_3_-54",
            "ap_3_-57", "f107_1944_0", "f107_24_-24"
        ]
        
        # 加载CSV数据
        self.data = self._load_config()
        
        # ✅ 新增：加载所有可用日期
        self.available_dates = self._load_all_dates()
        
        # ✅ 新增：获取当前 Episode 对应的日期索引
        episode_idx = get_next_episode_date_index()
        date_idx = episode_idx % len(self.available_dates)  # 循环使用
        start_year, start_month, start_day = self.available_dates[date_idx]
        
        # 起始日期（仿真开始时的日期，对应 sim_time=0）
        self.start_year = start_year
        self.start_month = start_month
        self.start_day = start_day
        self.episode_idx = episode_idx
        self.date_idx = date_idx
        
        # 当前日期（用于 get_next_params 的兼容性）
        self.current_year = start_year
        self.current_month = start_month
        self.current_day = start_day
        self.current_index = 0
        
        print(f"[SpaceWeatherProvider] Episode #{episode_idx}, Date Index #{date_idx}/{len(self.available_dates)}")
        print(f"[SpaceWeatherProvider] Start date: {start_year}-{start_month:02d}-{start_day:02d}")
        print(f"[SpaceWeatherProvider] Interpolation enabled: linear between daily values")
    
    def _load_all_dates(self) -> list:
        """✅ 从日期配置文件读取所有日期信息（支持多个日期轮换）"""
        if self.date_config_file is None or not os.path.exists(self.date_config_file):
            default = [(1957, 10, 1)]
            print(f"[SpaceWeatherProvider] Date config file not found, using default: {default}")
            return default
        
        try:
            dates = []
            with open(self.date_config_file, "r", encoding="utf-8") as f:
                f.readline()  # 跳过header
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # 跳过空行和注释
                        parts = line.split(",")
                        try:
                            year = int(parts[0].strip())
                            month = int(parts[1].strip())
                            day = int(parts[2].strip())
                            dates.append((year, month, day))
                        except (ValueError, IndexError):
                            continue
            
            if dates:
                print(f"[SpaceWeatherProvider] Loaded {len(dates)} dates from config file")
                return dates
        except Exception as e:
            print(f"[SpaceWeatherProvider] Error loading date config: {e}")
        
        return [(1957, 10, 1)]
    
    def _load_date_config(self) -> tuple:
        """从日期配置文件读取起始日期信息（兼容旧接口）"""
        dates = self._load_all_dates()
        return dates[0]  # 返回第一个日期
        
        return (1957, 10, 1)
    
    def _load_config(self) -> list:
        """从CSV文件读取空间天气参数"""
        if self.config_file is None or not os.path.exists(self.config_file):
            print(f"[SpaceWeatherProvider] Config file not found: {self.config_file}")
            return []
        
        try:
            data = []
            with open(self.config_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        year = int(row["year"])
                        month = int(row["month"])
                        day = int(row["day"])
                    except (ValueError, KeyError) as e:
                        continue
                    
                    param_dict = {}
                    for key in self.sw_keys:
                        if key in row:
                            try:
                                value = row[key]
                                if value is None or value == '' or value.strip() == '':
                                    param_dict[key] = 8.0 if key.startswith("ap") else 110.0
                                else:
                                    param_dict[key] = float(value)
                            except (ValueError, TypeError):
                                param_dict[key] = 8.0 if key.startswith("ap") else 110.0
                        else:
                            param_dict[key] = 8.0 if key.startswith("ap") else 110.0
                    
                    date_key = (year, month, day)
                    self.data_by_date[date_key] = param_dict
                    data.append(param_dict)
            
            print(f"[SpaceWeatherProvider] Loaded {len(data)} records from {self.config_file}")
            return data
        except Exception as e:
            print(f"[SpaceWeatherProvider] Error loading config: {e}")
            return []
    
    def _date_to_tuple(self, year: int, month: int, day: int) -> tuple:
        """日期转换为元组"""
        return (year, month, day)
    
    def _get_next_date(self, year: int, month: int, day: int) -> tuple:
        """获取下一天的日期"""
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        # 处理闰年
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_month[1] = 29
        
        day += 1
        if day > days_in_month[month - 1]:
            day = 1
            month += 1
            if month > 12:
                month = 1
                year += 1
        
        return (year, month, day)
    
    def _add_days_to_date(self, year: int, month: int, day: int, num_days: int) -> tuple:
        """给日期加上指定天数"""
        for _ in range(num_days):
            year, month, day = self._get_next_date(year, month, day)
        return (year, month, day)
    
    def get_params_by_date(self, year: int, month: int, day: int) -> dict:
        """获取指定日期的参数"""
        date_key = (year, month, day)
        if date_key in self.data_by_date:
            return self.data_by_date[date_key]
        else:
            print(f"[SpaceWeatherProvider] No data for {year}-{month:02d}-{day:02d}, using default values")
            return {key: (8.0 if key.startswith("ap") else 110.0) for key in self.sw_keys}
    
    def get_interpolated_params(self, sim_time_seconds: float) -> dict:
        """
        根据仿真时间进行线性插值，获取当前时刻的空间天气参数。
        
        Args:
            sim_time_seconds: 仿真时间（秒），从仿真开始（对应起始日期零点）算起
        
        Returns:
            插值后的参数字典
        
        插值逻辑：
            - sim_time=0 对应 start_date 零点
            - sim_time=86400 对应 start_date+1 零点
            - 中间时刻进行线性插值
        """
        # 计算从起始日期零点开始经过的完整天数和当天内的时间比例
        elapsed_days = int(sim_time_seconds // self.SECONDS_PER_DAY)
        day_fraction = (sim_time_seconds % self.SECONDS_PER_DAY) / self.SECONDS_PER_DAY
        
        # 计算当天日期（sim_time 所在的那一天零点）
        today_date = self._add_days_to_date(
            self.start_year, self.start_month, self.start_day, elapsed_days
        )
        
        # 计算明天日期
        tomorrow_date = self._get_next_date(*today_date)
        
        # 获取今天和明天的参数
        today_params = self.get_params_by_date(*today_date)
        tomorrow_params = self.get_params_by_date(*tomorrow_date)
        
        # 线性插值：result = today * (1 - fraction) + tomorrow * fraction
        interpolated = {}
        for key in self.sw_keys:
            v_today = today_params.get(key, 8.0 if key.startswith("ap") else 110.0)
            v_tomorrow = tomorrow_params.get(key, 8.0 if key.startswith("ap") else 110.0)
            interpolated[key] = v_today * (1.0 - day_fraction) + v_tomorrow * day_fraction
        
        return interpolated
    
    def get_current_date_from_sim_time(self, sim_time_seconds: float) -> tuple:
        """根据仿真时间获取当前日期（用于日志显示）"""
        elapsed_days = int(sim_time_seconds // self.SECONDS_PER_DAY)
        return self._add_days_to_date(
            self.start_year, self.start_month, self.start_day, elapsed_days
        )

    
    def _advance_date(self):
        """推进当前日期到下一天"""
        self.current_year, self.current_month, self.current_day = self._get_next_date(
            self.current_year, self.current_month, self.current_day
        )
    
    def get_current_params(self) -> dict:
        """获取当前日期参数（不推进）"""
        return self.get_params_by_date(self.current_year, self.current_month, self.current_day)
    
    def reset(self):
        """
        重置并选择新的起始日期（用于 episode 重置）
        每次reset时轮流选择available_dates中的下一个日期，确保训练多样性
        """
        # ✅ 初始化本地episode计数器（如果不存在）
        if not hasattr(self, 'local_episode_count'):
            self.local_episode_count = 0
        
        # ✅ 轮流选择下一个日期
        date_idx = self.local_episode_count % len(self.available_dates)
        new_date = self.available_dates[date_idx]
        
        # ✅ 更新起始日期和当前日期
        self.start_year, self.start_month, self.start_day = new_date
        self.current_year = self.start_year
        self.current_month = self.start_month
        self.current_day = self.start_day
        self.current_index = 0
        
        self.local_episode_count += 1
        
        print(f"[SpaceWeatherProvider] Reset to new date: {self.start_year}-{self.start_month:02d}-{self.start_day:02d} "
              f"(episode #{self.local_episode_count}, date_idx={date_idx}/{len(self.available_dates)})")
# ==========================================
# 自定义环境模型
# ==========================================
class ExChangeWorld(GroundStationWorldModel):  
    def setup_atmosphere_density_model(self,**kwargs) -> None:
        """
        创建 MSIS 大气模型，并保存消息对象引用以便后续动态更新。
        """
        self.densityModel = msisAtmosphere.MsisAtmosphere()
        self.densityModel.ModelTag = "msisDensity"
        
        # 初始空间天气参数
        ap = 8
        f107 = 110
        
        # 显式定义通道顺序（与 MSIS 端口顺序匹配）
        sw_keys = [
            "ap_24_0", "ap_3_0", "ap_3_-3", "ap_3_-6", "ap_3_-9",
            "ap_3_-12", "ap_3_-15", "ap_3_-18", "ap_3_-21", "ap_3_-24",
            "ap_3_-27", "ap_3_-30", "ap_3_-33", "ap_3_-36", "ap_3_-39",
            "ap_3_-42", "ap_3_-45", "ap_3_-48", "ap_3_-51", "ap_3_-54",
            "ap_3_-57", "f107_1944_0", "f107_24_-24"
        ]
        
        # 创建并保存消息对象（用于后续动态更新）
        self.swMsgList = []
        self.swPayloadList = []  # 保存 payload 引用便于修改
        
        for idx, key in enumerate(sw_keys):
            val = ap if key.startswith("ap") else f107
            
            # 创建 payload 和消息对象
            payload = messaging.SwDataMsgPayload()
            payload.dataValue = val
            msg = messaging.SwDataMsg()
            msg.write(payload)
            
            # 保存引用
            self.swMsgList.append(msg)
            self.swPayloadList.append(payload)
            
            # 订阅到 MSIS 输入端口
            if idx < len(self.densityModel.swDataInMsgs):
                self.densityModel.swDataInMsgs[idx].subscribeTo(msg)
            else:
                print(f"[ExChangeWorld] Warning: MSIS port {idx} does not exist")
        
        # 订阅行星位置
        if hasattr(self, "gravFactory") and getattr(self.gravFactory, "spiceObject", None):
            try:
                self.densityModel.planetPosInMsg.subscribeTo(
                    self.gravFactory.spiceObject.planetStateOutMsgs[self.body_index]
                )
            except Exception as e:
                print(f"[ExChangeWorld] Warning subscribing planetPosInMsg: {e}")
        else:
            print("[ExChangeWorld] Warning: gravFactory not ready, planetPosInMsg not subscribed")
        
        # 添加到仿真任务
        self.simulator.AddModelToTask(self.world_task_name, self.densityModel, ModelPriority=1000)
        # print(f"[ExChangeWorld] MSIS atmosphere initialized with ap={ap}, f107={f107}")
    
    def update_space_weather(self, param_dict: dict):
        """
        动态更新空间天气参数。
        
        Args:
            param_dict: 包含所有21个sw_keys的字典
                       如 {"ap_24_0": 8.0, "ap_3_0": 10.0, ..., "f107_1944_0": 110.0}
        """
        if not hasattr(self, 'swMsgList'):
            print("[ExChangeWorld] Warning: swMsgList not initialized")
            return
        
        # 定义通道顺序（与初始化时一致）
        sw_keys = [
            "ap_24_0", "ap_3_0", "ap_3_-3", "ap_3_-6", "ap_3_-9",
            "ap_3_-12", "ap_3_-15", "ap_3_-18", "ap_3_-21", "ap_3_-24",
            "ap_3_-27", "ap_3_-30", "ap_3_-33", "ap_3_-36", "ap_3_-39",
            "ap_3_-42", "ap_3_-45", "ap_3_-48", "ap_3_-51", "ap_3_-54",
            "ap_3_-57", "f107_1944_0", "f107_24_-24"
        ]
        
        # 更新所有通道的值（按顺序一一对应）
        for idx, key in enumerate(sw_keys):
            if idx < len(self.swMsgList):
                val = param_dict.get(key, 8.0 if key.startswith("ap") else 110.0)
                
                # 创建新 payload 并写入消息
                payload = messaging.SwDataMsgPayload()
                payload.dataValue = val
                self.swMsgList[idx].write(payload)
        
        # 可选：打印日志
        # print(f"[ExChangeWorld] Updated space weather with {len(param_dict)} parameters")

# ==========================================
class ExponentialDynModel(dyn.FullFeaturedDynModel):
    @classmethod
    def _requires_world(cls):
        return [ExChangeWorld]

    def setup_density_model(self) -> None:
        self.world.densityModel.addSpacecraftToModel(self.scObject.scStateOutMsg)

    @default_args(dragCoeff=2.2)
    def setup_drag_effector(
        self, width: float = 1.0, depth: float = 1.0, height: float = 1.0,
        panelArea: float = 10.0, dragCoeff: float = 2.2, priority: int = 999, **kwargs
    ) -> None:
        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        self.dragEffector.addFacet(width * depth, dragCoeff, [1, 0, 0], [height / 2, 0.0, 0])
        self.dragEffector.addFacet(width * depth, dragCoeff, [-1, 0, 0], [height / 2, 0.0, 0])
        self.dragEffector.addFacet(height * width, dragCoeff, [0, 1, 0], [0, depth / 2, 0])
        self.dragEffector.addFacet(height * width, dragCoeff, [0, -1, 0], [0, -depth / 2, 0])
        self.dragEffector.addFacet(height * depth, dragCoeff, [0, 0, 1], [0, 0, width / 2])
        self.dragEffector.addFacet(height * depth, dragCoeff, [0, 0, -1], [0, 0, -width / 2])
        self.dragEffector.addFacet(panelArea / 2, dragCoeff, [0, 1, 0], [0, height, 0])
        self.dragEffector.addFacet(panelArea / 2, dragCoeff, [0, -1, 0], [0, height, 0])
        self.dragEffector.atmoDensInMsg.subscribeTo(self.world.densityModel.envOutMsgs[-1])
        self.scObject.addDynamicEffector(self.dragEffector)
        self.simulator.AddModelToTask(self.task_name, self.dragEffector, ModelPriority=priority)

    def setup_gravity_bodies(self) -> None:  # 配置点质量重力场（成功）
        """
        使用点质量模型来配置重力场。
        """
        super().setup_gravity_bodies()
        # 1. 初始化重力体工厂 (GravBodyFactory)
        gravFactory = simIncludeGravBody.gravBodyFactory()
        # 2. 创建地球的点质量模型 (默认即为 Point Mass)
        # 你也可以添加 'moon', 'sun' 等，只需调用对应的方法
        earth = gravFactory.createEarth()
        # earth.spherHarm.maxDeg = 0
        # 3. 设置为中心天体 (Central Body)
        # 这确保了仿真在相对于地球的坐标系中进行，提高数值精度
        earth.isCentralBody = True 
        # earth.useSphericalHarmParams = False  # 禁用球谐参数
        # 4. 将重力体工厂连接到卫星动力学模型
        # 这一步将重力加速度计算集成到卫星的运动方程中
        gravFactory.addBodiesTo(self.scObject)
        self.gravFactory = gravFactory
        # print(f"[ExponentialDynModel] Using Point Mass gravity model for Earth")
        # print(f"  μ_Earth = {earth.mu:.6e} m³/s²")
        # print(f"  Central Body: {earth.isCentralBody}")

# ==========================================
# 观测函数（修改：使用可配置的目标）
# ==========================================


def rv2a_deviation(sat) -> np.ndarray:

    r = np.array(sat.dynamics.r_BN_N)
    v = np.array(sat.dynamics.v_BN_N)
    
    current_a_km, is_valid = _calculate_sma(r, v)
    reference_a_km = sat._target_sma_km
    deviation = current_a_km - reference_a_km  # 使用真实值（不使用滤波值）
    
    # 检查 NaN/Inf
    if not np.isfinite(deviation):
        print(f"[WARN] rv2a_deviation: Invalid deviation={deviation}, returning 0.0")
        deviation = 0.0
    
    print(f"rv2a_deviation: current_a_km={current_a_km:.2f}, reference_a_km={reference_a_km:.2f}, deviation={deviation:.2f}")
    return np.array([deviation], dtype=np.float32)

def rv2a(sat) -> np.ndarray:
    return np.array([_calculate_sma(sat.dynamics.r_BN_N, sat.dynamics.v_BN_N)[0]])
def fuel_remaining(sat) -> np.ndarray:
    # print("sat.fsw.dv_available:", sat.fsw.dv_available)
    fuel = sat.fsw.dv_available
    if not np.isfinite(fuel):
        fuel = 480.0
    return np.array([fuel], dtype=np.float32)

def a(sat) -> np.ndarray:
    """计算高度 [km]"""
    r = np.linalg.norm(np.array(sat.dynamics.r_BN_N)) / 1000.0
    if not np.isfinite(r):
        r = REQ_EARTH + Target_sma  # 默认值
    return np.array([r], dtype=np.float32)
def v(sat) -> np.ndarray:
    """计算轨道速度偏差 [km/s]"""
    v = np.linalg.norm(np.array(sat.dynamics.v_BN_N)) / 1000.0
    v_init = V_init
    return np.array([v - v_init], dtype=np.float32)

def thrust_force(sat) -> np.ndarray:
    """获得推力观测值"""
    return np.array([float(getattr(sat, "thrust", 0.0))], dtype=np.float32)

def thrust_theta(sat) -> np.ndarray:
    """获得推力极角观测值"""
    return np.array([float(getattr(sat, "theta", 0.0))], dtype=np.float32)

def ap(sat) -> np.ndarray:
    """获得当前 ap 值"""
    print("ap value:", float(getattr(sat, "ap", 0.0)))
    return np.array([float(getattr(sat, "ap", 0.0))], dtype=np.float32)

def f107(sat) -> np.ndarray:
    """获得当前 f107 值"""
    print("f107 value:", float(getattr(sat, "f107", 0.0)))
    return np.array([float(getattr(sat, "f107", 0.0))], dtype=np.float32)

def atmospheric_density(sat) -> np.ndarray:
    """获得卫星当前位置的大气密度 [kg/m^3]"""
    try:
        # 从 dragEffector 读取（它订阅了 MSIS 的输出）
        if hasattr(sat.dynamics, 'dragEffector') and sat.dynamics.dragEffector is not None:
            msg_reader = sat.dynamics.dragEffector.atmoDensInMsg
            if msg_reader.isWritten():
                payload = msg_reader()  # 直接调用 reader
                density = float(payload.neutralDensity)
            else:
                density = 0.0
        else:
            density = 0.0
    except Exception as e:
        print(f"Error getting atmospheric density: {e}")
        density = 0.0
    return np.array([density], dtype=np.float32)

# ==========================================
# 根据霍曼转移轨道公式，计算推力动作
# ==========================================
"""
动作的输入只有两个个变量：提升或降低的高度dH
在动作类内部，创建一个hohmann转移函数，函数从环境中获得卫星装填信息，包括卫星的高度，速度，远地点角，近地点角，真近地点角等等
根据这些信息，计算出需要的推力矢量，在hill坐标系下计算推力矢量，然后施加推力
"""

class ImpulsiveThrustHillFixedTime(act.ImpulsiveThrust):  #
    def __init__(
        self, 
        chief_name: str, 
        fixed_duration: float, 
        *args, 
        **kwargs
    ):
        """
        在 Hill 坐标系下施加脉冲推力，具有固定漂移时间和决策阈值。
        转化为极坐标
        Args:
            chief_name: 参考卫星名称（选自己则填自己名字）。
            fixed_duration: 固定的漂移/执行持续时间 [s]。
            threshold: 激活推力的决策阈值，默认为 0.5。
            *args, **kwargs: 传递给 ImpulsiveThrust 的参数（如 max_dv, name）。
        """
        self.chief_name = chief_name
        self.fixed_duration = fixed_duration
        # self.threshold = threshold
        # self.max_theta = max_theta  # 极角范围
        # self.max_beta = max_beta    # 方位角范围
        super().__init__(*args, **kwargs)

    @property
    def space(self) -> spaces.Box:
        """
        重新定义动作空间：
        [dv, theta]
        前三个是极坐标系下的速度，两个角度参数，最后一个是决策概率。
        """
        return spaces.Box(
            low=np.array([-self.max_dv]),
            high=np.array([self.max_dv]),
            shape=(1,),
            dtype=np.float32,
        )
    @property
    def action_description(self) -> list[str]:
        return ["dV", "threashold"]
    def reset_post_sim_init(self) -> None:
        self.chief = self.satellite.simulator.get_satellite(self.chief_name)

    def set_action(self, action: np.ndarray) -> None:
        
        assert len(action) == 1, "Action must have 1 element: [dV]"
        dv = action[0]
        
        # 极坐标转换（转换为hill坐标系）
        dv_Hill_R = 0.0
        dv_Hill_S = dv
        dv_Hill_W = 0.0
        #------------------------------------
        dv_H = [dv_Hill_R, dv_Hill_S, dv_Hill_W]
        # 这里的 NH 是从惯性系到 Hill 的转置，即 Hill 到惯性
        NH = self.chief.dynamics.HN.T
        dv_N = NH @ dv_H
        
        # 3. 最大推力限制 (调用基类的逻辑进行 clamping)
        dv_mag = np.linalg.norm(dv_N)
        if dv_mag > self.max_dv:
            dv_N = dv_N / dv_mag * self.max_dv
        
        self.satellite.logger.info(f"Executing 极坐标 thrust: {np.linalg.norm(dv_N)} m/s ")
        # 记录到卫星属性，供观测/日志/奖励读取
        setattr(self.satellite, "thrust", np.linalg.norm(dv_N))
        # setattr(self.satellite, "theta", float(action[1]))

        # 4. 执行 FSW 动作
        self.satellite.fsw.action_impulsive_thrust(dv_N)
        
        # 5. 使用固定的持续时间更新模拟停止时间
        self.satellite.update_timed_terminal_event(
            self.satellite.simulator.sim_time + self.fixed_duration
        )

        # 6. 激活可选的 FSW 漂移模式（如指向模式）
        if self.fsw_action is not None:
            getattr(self.satellite.fsw, self.fsw_action)()

class HohmannThrust(act.ImpulsiveThrust):
    def __init__(
        self, 
        chief_name: str, 
        max_dh: float = 10.0,   # 最大高度变化 m
        *args, 
        **kwargs
    ):
        """
        在 Hill 坐标系下施加脉冲推力，具有固定漂移时间和决策阈值。
        转化为极坐标
        Args:
            chief_name: 参考卫星名称（选自己则填自己名字）。
            fixed_duration: 固定的漂移/执行持续时间 [s]。
            *args, **kwargs: 传递给 ImpulsiveThrust 的参数（如 max_dv, name）。
            max_dh: 最大高度变化 [km]。
        """
        self.chief_name = chief_name
        self.max_dh = max_dh
        super().__init__(*args, **kwargs)
    
    @property
    def space(self) -> spaces.Box:
        """
        重新定义动作空间：
        [dh]
        第一个是高度变化
        """
        return spaces.Box(
            low=np.array([-self.max_dh]),
            high=np.array([self.max_dh]),
            shape=(1,),
            dtype=np.float32,
        )
    
    def reset_post_sim_init(self) -> None:
        self.chief = self.satellite.simulator.get_satellite(self.chief_name)
        # 清理旧的第二次机动事件（如果存在）
        # 由于我们现在使用时间戳，旧事件会自然失效，无需显式清理
    
    def hohmann_transfer(self, delta_sma: float, current_sma: float):
        """
        计算从半径 r1 到 r2 的 Hohmann 转移所需的速度变化（Δv）和转移时间

        参数:
            r1 (float): 初始圆轨道半径，单位为 km
            r2 (float): 目标圆轨道半径，单位为 km
            mu (float): 中心天体的标准引力参数，单位为 km^3/s^2，默认为地球的 μ ≈ 398600.4418 km^3/s^2

        返回:
            dv1 (float): 第一次推进所需的速度变化（km/s）
            dv2 (float): 第二次推进所需的速度变化（km/s）
            delta_v_total (float): 总的速度变化（km/s）
            t_transfer (float): 转移时间（秒）
        """
        mu = MU_EARTH * 1e9  # 转换为 m^3/s^2
        target_sma = current_sma + delta_sma # delta_sma单位为m
        r1 = current_sma # 输入为m
        r2 = target_sma  # 单位为m
        # 初始和目标轨道速度
        v1 = np.sqrt(mu / r1) # 单位为 m/s
        v2 = np.sqrt(mu / r2) # 单位为 m/s
        # print(f"v1: {v1}, v2: {v2}")
        # 转移轨道半长轴
        a_transfer = 0.5 * (r1 + r2)
        # 转移轨道速度
        v_transfer1 = np.sqrt(mu * (2/r1 - 1/a_transfer)) # 单位为 m/s
        v_transfer2 = np.sqrt(mu * (2/r2 - 1/a_transfer)) # 单位为 m/s
        # Δv
        dv1 = v_transfer1 - v1 # 单位为 m/s
        dv2 = v2 - v_transfer2 # 单位为 m/s
        # 转移时间（半个椭圆轨道周期）
        t_transfer = np.pi * np.sqrt(a_transfer**3 / mu) # 单位为 s
        
        t_wait = np.pi * np.sqrt((target_sma**3) / mu) # 等待半圈时间，单位为 s
        dv_action = np.array([dv1, dv2, t_transfer, t_wait])
        return dv_action

    def set_action(self, action: np.ndarray) -> None:
        """
        获取卫星位置信息, 传递给hohmann_transfer()
        调用hohmann_transfer()，输出推力矢量
        将推力矢量，转换为惯性系，传递给仿真器
        卫星执行动作
        """
        assert len(action) == 1, "Action must have 1 element: [dh]"

        r = self.satellite.dynamics.r_BN_N
        v = self.satellite.dynamics.v_BN_N
        r = np.array(r)
        v = np.array(v)
        NH = self.chief.dynamics.HN.T
        dh = action[0].item()  # 高度变化，单位为 m
        # 1.计算霍曼转移所需的推力矢量
        try:
            oe = orbitalMotion.rv2elem(mu=MU_EARTH * 1e9, rVec=r, vVec=v)  # 单位为m，使用当前半长轴
            if oe is None:
                print("[WARN] set_action: rv2elem returned None, using default SMA")
                current_sma_m = (REQ_EARTH + Target_sma) * 1000  # 默认值，单位为m
            else:
                current_sma_m = float(oe.a)  # 单位为m
        except Exception as e:
            print(f"[WARN] set_action: rv2elem exception {e}, using default SMA")
            current_sma_m = (REQ_EARTH + Target_sma) * 1000  # 默认值，单位为m
        print(f"current_sma_m: {current_sma_m:.2f} m, current_sma_km: {current_sma_m/1000:.2f} km")
        dv_action = self.hohmann_transfer(dh, current_sma_m)
        dv_1 = dv_action[0].item()
        dv_2 = dv_action[1].item()
        self.t_transfer = dv_action[2].item()
        self.t_wait = dv_action[3].item()
        # 2. 第一次动作，转换为 Hill 坐标系下的推力矢量
        dv_H_1 = [0.0, dv_1, 0.0]
        # 这里的 NH 是从惯性系到 Hill 的转置，即 Hill 到惯性
        dv_N_1 = NH @ dv_H_1
        # 执行第一次机动（立即，在t=0）
        self.satellite.logger.info(f"第一次机动 Executing thrust: {np.linalg.norm(dv_N_1):.3f} m/s ")
        self.satellite.fsw.action_impulsive_thrust(dv_N_1)
        
        # 3. 创建第二次机动的延迟事件（在t=t_transfer时执行）
        from Basilisk.utilities import macros
        
        # 保存第二次机动的参数
        dv_H_2 = [0.0, dv_2, 0.0]
        
        # 定义第二次机动的执行函数
        def execute_second_burn(sim_object):
            """在t_transfer时刻执行第二次Hohmann机动"""
            # 重新获取Hill坐标系（姿态可能已变化）
            NH_current = self.chief.dynamics.HN.T
            dv_N_2 = NH_current @ np.array(dv_H_2)
            
            self.satellite.logger.info(
                f"第二次机动 (t={self.satellite.simulator.sim_time:.1f}s): "
                f"dv={np.linalg.norm(dv_N_2):.3f} m/s"
            )
            self.satellite.fsw.action_impulsive_thrust(dv_N_2)
        
        # 在Basilisk中创建延迟事件
        second_burn_time = self.satellite.simulator.sim_time + self.t_transfer
        # 使用时间戳确保事件名称唯一
        event_name = f"hohmann_second_burn_{self.satellite.name}_{int(second_burn_time*1000)}"
        self.satellite.simulator.createNewEvent(
            event_name,
            macros.sec2nano(self.satellite.simulator.sim_rate),
            True,  # eventActive
            conditionTime=macros.sec2nano(second_burn_time),
            actionFunction=execute_second_burn,
            terminal=False,  # 非终止事件，仿真继续
        )
        
        self.satellite.logger.info(
            f"已调度第二次机动在 t={second_burn_time:.1f}s "
            f"(转移时间={self.t_transfer:.1f}s)"
        )
        
        # 5. 设置总的仿真停止时间（转移+等待）
        total_time = self.t_transfer + self.t_wait
        self.satellite.update_timed_terminal_event(
            self.satellite.simulator.sim_time + total_time
        )
        
        self.satellite.logger.info(
            f"仿真将运行 {total_time:.1f}s (转移{self.t_transfer:.1f}s + 等待{self.t_wait:.1f}s)"
        )
        # 6. 激活可选的 FSW 漂移模式
        if self.fsw_action is not None:
            getattr(self.satellite.fsw, self.fsw_action)()

# ==========================================
# 自定义卫星类（修改：存储目标值）
# ==========================================

class MySatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="rv2a_deviation", fn=rv2a_deviation, norm=1.0),
            dict(prop="sma_km", fn=rv2a, norm=8000.0),
            dict(prop="fuel_remaining", fn=fuel_remaining, norm=500.0),
            # dict(prop="ap", fn=ap, norm=100.0),  # 控制效果与ap的相关性大，可能是归一化程度不够
            # dict(prop="f107", fn=f107, norm=300.0),
            # dict(prop="atmospheric_density", fn=atmospheric_density, norm=1e-10),  # 新增：大气密度观测
        ),
    ]   
    action_spec = [  # 使用自定义动作
        HohmannThrust(
            chief_name = "EO1",
            max_dh=Max_Hohmann_dh,
            fsw_action=None,
        ),
    ] 
    dyn_type = ExponentialDynModel 
    fsw_type = fsw.MagicOrbitalManeuverFSWModel

    def __init__(self, *args, target_sma_km=None, **kwargs):
        """
        Args:
            target_sma_km: 目标轨道半长轴 [km]（包含地球半径）
                          如果为 None，使用 orbitalMotion.REQ_EARTH + Target_sma
        """
        super().__init__(*args, **kwargs)
        self._target_sma_km = target_sma_km

# ==========================================
# 环境包装器（修改：添加调试信息）
# ==========================================

class DensityWrapper(gym.Wrapper):
    def __init__(
        self, env: gym.Env,
        density_schedule: Optional[Callable[[float], float]] = None,
        space_weather_provider: Optional[SpaceWeatherProvider] = None,
        target_alt_km: float = Target_sma,
        initial_alt: float = None,
    ):
        super().__init__(env)
        self.density_schedule = density_schedule
        self.space_weather_provider = space_weather_provider
        self.step_count = 0
        self._density_model = None
        self._world_model = None
        self.pid = os.getpid()
        self.episode_count = 0
        self._target_alt_km = target_alt_km
        self._target_sma_km = REQ_EARTH + target_alt_km
        self._initial_alt = initial_alt
        
        # ✅ 用于记录上次更新时的日期（避免重复打印日志）
        self._last_logged_date = None

        if hasattr(self.unwrapped, 'satellite'):
            sat = self.unwrapped.satellite
            if hasattr(sat, '_target_sma_km'):
                print(f"[DensityWrapper PID={self.pid}] Satellite target SMA: {sat._target_sma_km:.2f} km")
        if hasattr(self.unwrapped, 'rewarder'):
            rew = self.unwrapped.rewarder
            if hasattr(rew, 'target_sma_km'):
                print(f"[DensityWrapper PID={self.pid}] Rewarder target SMA: {rew.target_sma_km:.2f} km")
    
    @property
    def density_model(self):
        if self._density_model is None:
            self._density_model = self.unwrapped.satellite.dynamics.world.densityModel
        return self._density_model
    
    @property
    def world_model(self):
        if self._world_model is None:
            self._world_model = self.unwrapped.satellite.dynamics.world
        return self._world_model
    
    def _get_sim_time(self) -> float:
        return float(getattr(self.unwrapped.simulator, "sim_time", 0.0))
    
    def _log_state(self, reward=0.0):
        try:
            r_BN_N = np.array(self.unwrapped.satellite.dynamics.r_BN_N)
            v_BN_N = np.array(self.unwrapped.satellite.dynamics.v_BN_N)
            alt_km = np.linalg.norm(r_BN_N) / 1000.0 - REQ_EARTH
            deviation_km = alt_km - self._target_alt_km
            semi_axis_km, _ = _calculate_sma(r_BN_N, v_BN_N)
            sim_time = self._get_sim_time()
            ref_sma_km = self._target_sma_km
            rv2a_deviation_km = semi_axis_km - ref_sma_km
            dv_available = self.unwrapped.satellite.fsw.dv_available
            r_km = r_BN_N / 1000.0
            v_km_s = v_BN_N / 1000.0
            v_magnitude = np.linalg.norm(v_BN_N) / 1000.0  # ✅ 改为计算v向量大小
            
            # ✅ 获取ap和f107值
            sat = self.unwrapped.satellite
            ap_val = float(getattr(sat, "ap", 8.0))
            f107_val = float(getattr(sat, "f107", 110.0))
            
            # 获取大气密度值（从 dragEffector 读取）
            try:
                if hasattr(sat.dynamics, 'dragEffector') and sat.dynamics.dragEffector is not None:
                    msg_reader = sat.dynamics.dragEffector.atmoDensInMsg
                    if msg_reader.isWritten():
                        payload = msg_reader()  # 直接调用 reader
                        atm_density = float(payload.neutralDensity)
                    else:
                        atm_density = 0.0
                else:
                    atm_density = 0.0
            except Exception as e:
                print(f"[_log_state] Error reading density: {e}")
                atm_density = 0.0
            
            with open(LOG_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.pid, self.step_count, 
                    getattr(self.density_model, 'baseDensity', 0.0),
                    alt_km, v_magnitude, deviation_km, semi_axis_km, 
                    rv2a_deviation_km, sim_time,
                    reward, dv_available, ap_val, f107_val, atm_density,  # ✅ 新增：大气密度
                ])
        except Exception as e:
            print(f"[DensityWrapper] Logging error: {e}")

    def reset(self, **kwargs):
        self.step_count = 0
        self._density_model = None
        self._world_model = None
        self.thrust_blocked_count = 0
        self.thrust_allowed_count = 0
        self._last_logged_date = None
        
        # ✅ 重置 SpaceWeatherProvider 到起始日期
        if self.space_weather_provider is not None:
            self.space_weather_provider.reset()
        
        # 每个 episode 更新卫星的目标高度
        if self.altitude_config_manager is not None:
            initial_alt, target_alt = self.altitude_config_manager.get_next_config(Target_sma)
            
            satellite = self.unwrapped.satellite
            satellite._target_sma_km = REQ_EARTH + target_alt
            oe = random_circular_orbit(alt=initial_alt, i=0.0, Omega=0.0, f=0.0)
            satellite.oe = oe
            rewarder = self.unwrapped.rewarder
            rewarder._target_sma_km = REQ_EARTH + target_alt
            satellite._target_sma_km = REQ_EARTH + target_alt
            rewarder.last_fuel = {}
            rewarder.last_time = {}
            
            self.episode_count += 1
            print(f"\n{'='*80}")
            print(f"[Episode {self.episode_count}] PID={self.pid}")
            print(f"  Initial Altitude: {initial_alt:.2f} km")
            print(f"  Target Altitude:  {target_alt:.2f} km")
            print(f"  Target SMA:       {satellite._target_sma_km:.2f} km")
            print(f"  Rewarder fuel_cost: {rewarder.fuel_cost if rewarder else 'N/A'}")
            print(f"  Rewarder dist_penalty_scale: {rewarder.dist_penalty_scale if rewarder else 'N/A'}")
            
            if self.space_weather_provider is not None:
                sw_date = f"{self.space_weather_provider.start_year}-{self.space_weather_provider.start_month:02d}-{self.space_weather_provider.start_day:02d}"
                print(f"  Space Weather Start Date: {sw_date}")
            print(f"{'='*80}\n")
        
        # ✅ 在调用 env.reset() 之前先初始化卫星的 ap 和 f107 属性
        # 这样 reset() 返回的观测值才会包含正确的空间天气参数
        if self.space_weather_provider is not None:
            param_dict = self.space_weather_provider.get_interpolated_params(0.0)
            sat = self.unwrapped.satellite
            setattr(sat, "ap", param_dict.get("ap_24_0", 8.0))
            setattr(sat, "f107", param_dict.get("f107_1944_0", 110.0))
        
        obs, info = self.env.reset(**kwargs)
        
        # ✅ reset 后再次更新空间天气和 MSIS 模型（sim_time=0）
        self._update_space_weather()
        self._log_state(reward=0.0)
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        action_array = np.array(action, dtype=np.float32, copy=True)

        obs, reward, terminated, truncated, info = self.env.step(action_array)
        
        # ✅ 检查观测值是否包含 NaN/Inf
        if isinstance(obs, np.ndarray):
            if not np.all(np.isfinite(obs)):
                print(f"[WARN] DensityWrapper: NaN/Inf detected in observation: {obs}")
                print("[WARN] DensityWrapper: Replacing NaN/Inf with zeros and terminating episode")
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                terminated = True
        elif isinstance(obs, dict):
            for key, val in obs.items():
                if isinstance(val, np.ndarray) and not np.all(np.isfinite(val)):
                    print(f"[WARN] DensityWrapper: NaN/Inf detected in observation[{key}]: {val}")
                    obs[key] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                    terminated = True
        
        # ✅ 检查轨道计算是否有效，如果无效则终止episode
        if hasattr(self.unwrapped.satellite.data_store, 'reward') and \
           hasattr(self.unwrapped.satellite.data_store.reward, 'orbit_invalid') and \
           self.unwrapped.satellite.data_store.reward.orbit_invalid:
            print("[WARN] DensityWrapper: Orbit calculation invalid, terminating episode")
            terminated = True
        
        # ✅ 根据仿真时间插值更新空间天气
        self._update_space_weather()
        
        print("reward_shape: ", np.shape(reward))
        reward_value = sum(reward.values()) / len(reward) if isinstance(reward, dict) else float(reward)
        self._log_state(reward=reward_value)

        info = dict(info) if info is not None else {}
        
        # 添加燃料信息到info中，用于TensorBoard记录
        info["dv_available"] = float(self.unwrapped.satellite.fsw.dv_available)
        info["fuel_consumed"] = float(Max_fuel - self.unwrapped.satellite.fsw.dv_available)
        
        return obs, reward, terminated, truncated, info
    
    def _update_space_weather(self):
        """
        根据仿真时间进行线性插值，更新 MSIS 空间天气参数。
        
        插值逻辑：
            - sim_time=0 对应起始日期零点
            - sim_time=86400 对应起始日期+1天零点
            - 中间时刻按比例线性插值
        """
        if self.space_weather_provider is None:
            return
        
        try:
            # 获取当前仿真时间
            sim_time = self._get_sim_time()
            
            # ✅ 使用插值方法获取当前时刻的参数
            param_dict = self.space_weather_provider.get_interpolated_params(sim_time)
            
            # 更新 MSIS 模型
            self.world_model.update_space_weather(param_dict)
            
            # 更新卫星属性（供观测使用）
            sat = self.unwrapped.satellite
            ap_val = param_dict.get("ap_24_0", 8.0)
            f107_val = param_dict.get("f107_1944_0", 110.0)
            setattr(sat, "ap", ap_val)
            setattr(sat, "f107", f107_val)
            
            # ✅ 每 100 步打印一次，便于观察插值过程
            if self.step_count % 100 == 0:
                current_date = self.space_weather_provider.get_current_date_from_sim_time(sim_time)
                day_fraction = (sim_time % SpaceWeatherProvider.SECONDS_PER_DAY) / SpaceWeatherProvider.SECONDS_PER_DAY
                print(f"[DensityWrapper PID={self.pid}] Step={self.step_count}, "
                      f"Date={current_date[0]}-{current_date[1]:02d}-{current_date[2]:02d}, "
                      f"sim_time={sim_time:.1f}s, day_fraction={day_fraction:.4f}, "
                      f"ap={ap_val:.2f}, f107={f107_val:.2f}")
                
        except Exception as e:
            print(f"[DensityWrapper] Error updating space weather: {e}")
            import traceback
            traceback.print_exc()

# ==========================================
# 自定义TensorBoard回调
# ==========================================
class FuelLoggingCallbacks(WrappedEpisodeDataCallbacks):
    """自定义回调，用于在TensorBoard中记录燃料消耗等指标"""
    
    def on_episode_step(self, *, worker, base_env, policies, episode, **kwargs):
        """每步记录指标"""
        super().on_episode_step(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
        
        # 获取最新的info
        info = episode.last_info_for()
        if info and isinstance(info, dict):
            # 记录剩余燃料
            if "dv_available" in info:
                episode.user_data.setdefault("dv_available", []).append(info["dv_available"])
            
            # 记录消耗的燃料
            if "fuel_consumed" in info:
                episode.user_data.setdefault("fuel_consumed", []).append(info["fuel_consumed"])
    
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Episode结束时，计算统计指标并记录到TensorBoard"""
        super().on_episode_end(worker=worker, base_env=base_env, policies=policies, episode=episode, **kwargs)
        
        # 记录最终剩余燃料
        if "dv_available" in episode.user_data and len(episode.user_data["dv_available"]) > 0:
            final_fuel = episode.user_data["dv_available"][-1]
            episode.custom_metrics["final_dv_available"] = final_fuel
            episode.custom_metrics["mean_dv_available"] = np.mean(episode.user_data["dv_available"])
        
        # 记录总燃料消耗
        if "fuel_consumed" in episode.user_data and len(episode.user_data["fuel_consumed"]) > 0:
            total_fuel_consumed = episode.user_data["fuel_consumed"][-1]
            episode.custom_metrics["total_fuel_consumed"] = total_fuel_consumed
            episode.custom_metrics["mean_fuel_consumed"] = np.mean(episode.user_data["fuel_consumed"])

# ==========================================
# 环境配置
# ==========================================

gs_data = [
    dict(name="GS_Alaska", lat=64.0, long=-147.5, elev=0.0),
    dict(name="GS_Norway", lat=69.0, long=18.9, elev=0.0),
    dict(name="GS_Australia", lat=-35.3, long=149.1, elev=0.0),
]

env_args = dict(
    satellite=None,
    rewarder=None,  # ✅ 改为 None，在 make_wrapped_env 中创建
    failure_penalty = -5.0, # 卫星失败惩罚
    max_step_duration=5700 * 500, # 每一步的最大持续时间
    time_limit = 5700 * 500,
    log_level = "WARNING",
    world_args = dict(
        groundStationsData=gs_data,
        gsMinimumElevation=np.radians(10.0),
        gsMaximumRange=-1,
        utc_init="2018 SEP 29 21:00:00.000 (UTC)",
    ),
)

# 创建读取目标轨道和起始轨道参数
altitude_config_manager = AltitudeConfigManager(
    config_file="/workspace/learn_basilisk/RL-Hohmann-True/altitude_data.csv"
)

def make_wrapped_env(env_config):
    config = env_config.copy()
    space_weather_enabled = config.pop("space_weather_enabled", True)
    episode_callback = config.pop("episode_data_callback", None)
    default_alt = config.pop("target_altitude", Target_sma)

    initial_alt, target_alt = altitude_config_manager.get_next_config(default_alt)
    
    # ✅ 打印初始化参数
    print(f"\n{'='*80}")
    print(f"[make_wrapped_env] Environment created (PID={os.getpid()})")
    print(f"  Initial Altitude: {initial_alt:.2f} km")
    print(f"  Target Altitude:  {target_alt:.2f} km")
    print(f"{'='*80}\n")
    
    dist_penalty_scale = config.pop("dist_penalty_scale", 0.5)
    fuel_cost = config.pop("fuel_cost", 30)
    
    oe = random_circular_orbit(alt=initial_alt, i=0.0, Omega=0.0, f=0.0)
    sat_args = {"oe": oe, "mass": 300.0, "dv_available_init": Max_fuel,"batteryStorageCapacity": 200*3600.0}
    
    target_sma_km = REQ_EARTH + target_alt
    satellite = MySatellite(name="EO1", sat_args=sat_args, target_sma_km=target_sma_km)
    config["satellite"] = satellite
    
    rewarder = StationKeepingReward(
        satellite=satellite,
        target_alt_km=target_alt,
        dist_penalty_scale=dist_penalty_scale,
        fuel_cost=fuel_cost
    )
    config["rewarder"] = rewarder

    # ✅ 从日期配置文件读取空间天气参数
    sw_provider = SpaceWeatherProvider(
        config_file="/workspace/learn_basilisk/RL-Hohmann-True/basilisk_spaceweather.csv",
        date_config_file="/workspace/learn_basilisk/RL-Hohmann-True/date_config.txt"
    )
    base_env = SatelliteTasking(**config)
    
    wrapper = DensityWrapper(
        base_env,
        space_weather_provider=sw_provider,
    )
    
    # ✅ 传入管理器引用
    wrapper.altitude_config_manager = altitude_config_manager
    
    return wrapper

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

# ==========================================
# 网络搭建
# ==========================================

class MyContinuousActorCritic(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        obs_dim = obs_space.shape[0]
        act_dim = action_space.shape[0]

        # -------------------------
        # Actor network
        # -------------------------
        self.actor = nn.Sequential(
            nn.BatchNorm1d(obs_dim),
            nn.Linear(obs_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 450),
            nn.Tanh(),
            nn.Linear(450, act_dim),
            nn.Tanh()
        )

        # trainable log_std
        self.log_std = nn.Parameter(torch.ones(act_dim)* -1.0)

        # -------------------------
        # Critic network
        # -------------------------
        self.critic = nn.Sequential(
            nn.BatchNorm1d(obs_dim),
            nn.Linear(obs_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 450),
            nn.Tanh(),
            nn.Linear(450, 1)
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"].float()

        # Actor: output mean
        mean = self.actor(obs)

        # Critic: output V(s)
        self._value_out = self.critic(obs).squeeze(-1)

        # PPO expects concatenated [mean, log_std] for DiagGaussian distribution
        log_std = self.log_std.expand_as(mean)
        logits = torch.cat([mean, log_std], dim=-1)
        
        return logits, state

    def value_function(self):
        return self._value_out

# 注册模型
ModelCatalog.register_custom_model("my_model", MyContinuousActorCritic)


# ==========================================
# 训练配置（修改：启用空间天气）
# ==========================================
N_CPUS = 8
N_GPUS = 1
training_args = dict(
    lr=1e-5,  
    gamma=1,
    train_batch_size=2500,  # 训练批次大小，如果仿真很短的步数就结束了，会从头再次开始采样，直到达到训练批次大小
    # minibatch_size=512,
    num_sgd_iter=10,
    # model={"custom_model": "my_model"},
    model = dict(fcnet_hiddens=[128, 128], fcnet_activation="relu", vf_share_layers=False),
    lambda_=0.95,
    use_kl_loss=True,
    clip_param=0.3,
    grad_clip=0.3,  
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(
        num_env_runners=N_CPUS-1, 
        sample_timeout_s=1000.0,
    )
    .environment(
        env="SatelliteTasking-Density",
        env_config=dict(
            **env_args,
            episode_data_callback=episode_data_callback,
            space_weather_enabled=True,      # 启用动态空间天气
            target_altitude=Target_sma,
            dist_penalty_scale=1,
            fuel_cost=10,
        ),
    )
    # .resources(num_gpus=N_GPUS)
    .reporting(
        metrics_num_episodes_for_smoothing=1,
        metrics_episode_collection_timeout_s=180,
    )
    .checkpointing(export_native_model_files=True)
    .framework(framework="torch")
    .debugging(log_level="INFO")
    .callbacks(FuelLoggingCallbacks)
)   
# ==========================================
# 训练执行
# ==========================================
ray.init(
    ignore_reinit_error=True,
    num_cpus=N_CPUS,
    num_gpus=N_GPUS,
    object_store_memory=12_000_000_000,
)

CHECKPOINT_PATH = None  # 如果有检查点则填写路径，否则为 None
try:
    if CHECKPOINT_PATH:
        # tune.run 支持 restore 参数，传入检查点路径以恢复
        results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 2000},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        storage_path=RAY_LOG_FILE,
        verbose=1,
        restore=CHECKPOINT_PATH,
    )
    else:
        results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop={"training_iteration": 1000},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        storage_path=RAY_LOG_FILE,
        verbose=1,
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
finally:
    ray.shutdown()

# ==========================================
# 可视化
# ==========================================
if PLOT_RESULTS is True:
    print("Generating plots from", LOG_FILE)
    try:
        import pandas as pd
        from matplotlib.ticker import ScalarFormatter
        df = pd.read_csv(LOG_FILE)
        
        numeric_cols = [
            "step", "alt_km", "v_km_s","alt_deviation_km", "semi_axis_km",
            "rv2a_deviation_km",
            "sim_time", "reward", "dv_available", "ap", "f107", "atmospheric_density"  # 新增：大气密度
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        for pid in df["worker_pid"].unique():
            g = df[df["worker_pid"] == pid].sort_values("sim_time")
            fig, axes = plt.subplots(6, 1, figsize=(20, 24), sharex=True)
            
            axes[0].plot(g["sim_time"], g["alt_km"], alpha=0.8, label='Current Altitude', marker='o', markersize=2)
            axes[0].axhline(y=Target_sma, color='green', linestyle='--', linewidth=2, label=f'Target {Target_sma} km')
            axes[0].set_ylabel("Altitude (km)")
            axes[0].set_title(f"Worker {pid} - Satellite Altitude")
            axes[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[0].ticklabel_format(style='plain', axis='y')
            axes[0].legend()
            axes[0].grid(True)

            axes[1].plot(g["sim_time"], g["v_km_s"], alpha=0.4, label="Velocity", color='blue', marker='o', markersize=2)
            axes[1].set_ylabel("Velocity (km/s)")
            axes[1].set_title(f"Worker {pid} - Satellite Velocity")
            axes[1].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[1].ticklabel_format(style='plain', axis='y')
            axes[1].legend()
            axes[1].grid(True)


            axes[2].plot(g["sim_time"], g["semi_axis_km"], alpha=0.4, label="Raw SMA", color='blue', marker='o', markersize=2)
            target_sma_km = REQ_EARTH + Target_sma
            axes[2].axhline(y=target_sma_km, color='green', linestyle='--', linewidth=2, label=f'Target {target_sma_km:.1f} km')
            axes[2].set_ylabel("Semi-Major Axis (km)")
            axes[2].set_title(f"Worker {pid} - Semi-Major Axis")
            axes[2].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[2].ticklabel_format(style='plain', axis='y')
            axes[2].legend()
            axes[2].grid(True)
            
            axes[3].plot(g["sim_time"], g["rv2a_deviation_km"], alpha=0.4, label="Raw Deviation", color='blue', marker='o', markersize=2)
            axes[3].axhline(y=0, color='green', linestyle='--', linewidth=2, label='Target (0 km)')
            axes[3].set_ylabel("SMA Deviation (km)")
            axes[3].set_title(f"Worker {pid} - SMA Deviation from Target")
            axes[3].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[3].ticklabel_format(style='plain', axis='y')
            axes[3].legend()
            axes[3].grid(True)

            # ✅ 子图4：AP指数
            axes[4].plot(g["sim_time"], g["ap"], alpha=0.8, color='purple', linewidth=1.5, marker='o', markersize=2)
            axes[4].set_ylabel("AP Index", color='purple')
            axes[4].set_title(f"Worker {pid} - Space Weather: AP Index")
            axes[4].tick_params(axis='y', labelcolor='purple')
            axes[4].grid(True)
            axes[4].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[4].ticklabel_format(style='plain', axis='y')

            # ✅ 子图5：F107指数
            axes[5].plot(g["sim_time"], g["f107"], alpha=0.8, color='orange', linewidth=1.5, marker='o', markersize=2)
            axes[5].set_ylabel("F107 Index", color='orange')
            axes[5].set_xlabel("Simulation Time (s)")
            axes[5].set_title(f"Worker {pid} - Space Weather: F107 Index")
            axes[5].tick_params(axis='y', labelcolor='orange')
            axes[5].grid(True)
            axes[5].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            axes[5].ticklabel_format(style='plain', axis='y')
            
            out_png = f"./logs/worker_{pid}_timeseries.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"Plot saved to {out_png}")
    except Exception as e:
        print("Error during plotting:", e)
        import traceback
        traceback.print_exc()
