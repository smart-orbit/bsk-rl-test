import os
os.environ["RAY_DEDUP_LOGS"] = "0"  # 放在最开头，在 import ray 之前

from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import walker_delta_args
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
from bsk_rl.sim.world import BasicWorldModel

class ExChangeWorld(BasicWorldModel):

    @default_args(
        planetRadius=orbitalMotion.REQ_EARTH * 1e3,
        baseDensity=1.5,  # 修改默认海平面密度
        scaleHeight=8000,   # 修改默认尺度高度
    )
    def setup_atmosphere_density_model(
        self,
        planetRadius: float,
        baseDensity: float,
        scaleHeight: float,
        priority: int = 1000,
        **kwargs,
    ) -> None:
        # print("baseDensity:", baseDensity)
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

class ExponentialDynModel(dyn.ImagingDynModel):

    @classmethod
    def _requires_world(cls):
        """Specify that this dynamics model requires ExChangeWorld."""
        return [ExChangeWorld]

    def setup_density_model(self) -> None:
        # 调用 world 的 densityModel
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
        """设置拖曳效应器以响应大气密度变化"""

        """Set up the satellite drag effector.

        The drag effector causes aerodynamic forces and torques to act on the satellite.
        For purposes of this model, the satellite is assumed to be a rectangular prism
        with a solar panel on one end.

        Args:
            width: [m] Hub width.
            depth: [m] Hub depth.
            height: [m] Hub height.
            panelArea: [m^2] Solar panel surface area.
            dragCoeff: Drag coefficient.
            priority: Model priority.
            kwargs: Passed to other setup functions.
        """
        # print("dragCoeff:", dragCoeff)
        self.dragEffector = facetDragDynamicEffector.FacetDragDynamicEffector()
        self.dragEffector.ModelTag = "FacetDrag"
        #  Set up the geometry of a small satellite, starting w/ bus
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
        # Add solar panels
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


class MySatellite(sats.ImagingSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop='r_BN_N'),
            dict(prop="omega_BH_H", norm=0.03),  # 希尔坐标系中相对于希尔坐标系的物体角速度[rad/s]。
            # dict(prop="atmo_density", fn=lambda sat: sat.dynamics.update_atmosphere(random.randint(2,8))), 
        ),
        # obs.Eclipse()  # 观测返回入/出蚀时间（start, end）或两个数值
    ]
    action_spec = [  # 定义动作空间
        act.ImpulsiveThrust(name="DragThrust",
                            max_dv=1.0,
                            max_drift_duration=60.0,
                            fsw_action=None),  # Scan for 1 minute
        # act.Charge(duration=300),
    ] 
    dyn_type = ExponentialDynModel 
    fsw_type = fsw.MagicOrbitalManeuverFSWModel  

class DensityWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        density_schedule: Optional[Callable[[int], float]] = None,
    ):
        super().__init__(env)
        self.density_schedule = density_schedule
        self.step_count  = 0
        self._density_model = None
    
    @property
    def density_model(self):
        if self._density_model is None:
            self._density_model = self.unwrapped.satellite.dynamics.world.densityModel
        return self._density_model
    
    def reset(self, **kwargs):
        self.step_count = 0
        self._density_model = None  # Reset reference after env reset
        obs, info = self.env.reset(**kwargs)
        self._update_density()
        return obs, info
    
    def step(self, action):
        self._update_density()
        self.step_count += 1
        print(f"Step: {self.step_count}, Density: {self.density_model.baseDensity}")
        return self.env.step(action)
    
    def _update_density(self):
        self.density_model.baseDensity = self.density_schedule(self.step_count)
     
    
    def set_density(self, density: float):
        """Manually set density value."""
        self.density_model.baseDensity = density

def step_density(step: int) -> float:   # 输入int step,返回float baseDensity
    """根据步骤返回密度值"""

    # 函数中自己定义大气密度的变化
    if step < 5:
        return 1.0
    elif step < 10:
        return 2.0
    elif step < 15:
        return 3.0
    else:
        return 4.0

sat_arg_randomizer = walker_delta_args(altitude=250.0, inc=0.0, n_planes=1) 
MySatellite.default_sat_args()

sat_args = {}

# Make the satellite
sat = MySatellite(name="EO1", sat_args=sat_args)
targets = scene.UniformTargets(300)
env_args = dict(
    satellite=sat,
    scenario=targets,
    rewarder=data.NoReward(),  # 需要重写reward函数
    time_limit=5700.0 * 10,  # approximately 1 orbit
    log_level="WARNING",
    vizard_dir="/workspace/learn_basilisk/tmp",
    vizard_settings=dict(showLocationLabels=0),
    sat_arg_randomizer=sat_arg_randomizer,   

)

# env = gym.make(**env_args)

# 注册带 DensityWrapper 的环境
def make_wrapped_env(env_config):
    """创建带 DensityWrapper 的环境"""
    # 从 env_config 中提取 density_schedule
    # 使用 copy 避免修改原字典
    config = env_config.copy()
    
    # 从 config 中提取 density_schedule（如果有）
    density_schedule = config.pop("density_schedule", None)
    episode_callback = config.pop("episode_data_callback", None)
    
    # 创建基础环境
    base_env = SatelliteTasking(**config)
    
    # 如果提供了 density_schedule，则包装环境
    if density_schedule is not None:
        DensityWrapper(base_env, density_schedule=density_schedule)
    return base_env

# 注册自定义环境
register_env("SatelliteTasking-Density", make_wrapped_env)



# env = DensityWrapper(env, density_schedule=step_density)
# env.reset(seed = 1)
altitude = []

def episode_data_callback(env):
    reward = env.rewarder.cum_reward
    reward = sum(reward.values()) / len(reward)
    orbits = env.simulator.sim_time / (95 * 60)

    data = dict(
        reward=reward,
        # Are satellites dying, and how and when?
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

# 训练相关超参（按需调整）
N_CPUS = 8

training_args = dict(
    lr=0.00003,
    gamma=0.999,
    train_batch_size=250,  # usually a larger number, like 2500
    num_sgd_iter=10,
    model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False),
    lambda_=0.95,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=0.5,
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(num_env_runners=N_CPUS-1, sample_timeout_s=1000.0)
    .environment(
        env="SatelliteTasking-Density",      # 指定环境名称
        env_config=dict(**env_args,
        episode_data_callback=episode_data_callback,
        density_schedule=step_density),  # 环境参数，env_args 已定义，里面包含了卫星等东西
    )
    .reporting(
        metrics_num_episodes_for_smoothing=1,
        metrics_episode_collection_timeout_s=180,
    )
    .checkpointing(export_native_model_files=True)
    .framework(framework="torch")
    # .api_stack(
    #     enable_rl_module_and_learner=True,
    #     enable_env_runner_and_connector_v2=True,
    # )
    .callbacks(WrappedEpisodeDataCallbacks)
)   


ray.init(
    ignore_reinit_error=True,
    num_cpus=N_CPUS,
    object_store_memory=100_000_000  # 2 GB
)

# Run the training
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 2000},  # Adjust the number of iterations as needed
    checkpoint_freq=10,
    checkpoint_at_end=True,
    storage_path="/workspace/learn_basilisk/ray_results",
)

# Shutdown Ray
ray.shutdown()

# # 主循环中使用更大的密度值
# for i in range(25000):
#     action = np.array([0.0, 0.0, 0.0, 2.0])
#     observation, reward, terminated, truncated, info = env.step(action=action)
#     alt = (observation[0]**2 + observation[1]**2 + observation[2]**2)**0.5
#     altitude.append(alt)

# import pandas as pd
# import matplotlib.pyplot as plt

# plt.plot(np.arange(len(altitude)), altitude)
# plt.xlabel('Step')
# plt.ylabel('Altitude (m)')
# plt.title('Satellite Altitude Over Time with Atmospheric Drag')
# plt.grid()
# plt.savefig('satellite_altitude.png')