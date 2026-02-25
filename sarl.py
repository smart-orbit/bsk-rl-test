# 先创建一个有多卫星的环境

from bsk_rl import sats, act, obs, scene, data, comm
from bsk_rl.sim import dyn, fsw
from bsk_rl.utils.orbital import walker_delta_args
from bsk_rl import GeneralSatelliteTasking
from bsk_rl.utils.orbital import orbitalMotion, rv2HN
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from bsk_rl.utils.rllib.callbacks import WrappedEpisodeDataCallbacks
import ray
from ray import tune
from ray.tune.registry import register_env

# 新增：MSIS 大气和拖曳效应器
from Basilisk.simulation import msisAtmosphere, dragDynamicEffector

# 新增：基于 FullFeaturedDynModel 的自定义动力学模型，额外挂上 MSIS 大气
class MSISDynModel(dyn.FullFeaturedDynModel):
    """在 bsk_rl 现有动力学基础上，添加 MSIS 大气模型和对应的拖曳效应器。"""

    def reset_post_sim_init(self) -> None:
        """
        在 bsk_rl 创建好 Simulator、世界和飞行器之后调用。
        此时 self.simulator、self.scObject、self.task_name 都已经存在，适合挂接 Basilisk 模块。
        """
        # 先执行父类逻辑（保持原有行为）
        super().reset_post_sim_init()

        # 1) 创建 MSIS 大气模型
        # 类名是 MsisAtmosphere（可以通过 inspect 确认）
        self.msisAtmo = msisAtmosphere.MsisAtmosphere()
        self.msisAtmo.ModelTag = "MSISAtmosphere"
        epochMsg = unitTestSupport.timeStringToGregorianUTCMsg('2019 Jan 01 00:00:00.00 (UTC)')  # 设置大气模型的参考时间
        self.msisAtmo.epochInMsg.subscribeTo(epochMsg)
        # orbitalMotion.REQ_EARTH 是 [km]，转成 [m]
        self.msisAtmo.planetRadius = orbitalMotion.REQ_EARTH * 1e3

        # 如需更精细配置，可按 msisAtmosphere 源码设置太阳活动等参数
        # 例如：
        # self.msisAtmo.F107 = 150.0
        # self.msisAtmo.AP = 4.0

        # 将大气模型加入当前动力学任务
        self.simulator.AddModelToTask(self.task_name, self.msisAtmo)

        # 2) 创建拖曳效应器并订阅 MSIS 密度
        self.msisDragEff = dragDynamicEffector.DragDynamicEffector()  # 创建拖曳效应器实例
        self.msisDragEff.ModelTag = "MSISDrag"

        # DragDynamicEffector 的参数在 coreParams 结构下设置
        area = getattr(self, "dragArea", 2.0)
        cd = getattr(self, "dragCoeff", 2.2)
        self.msisDragEff.coreParams.area = area  # 阻力面积
        self.msisDragEff.coreParams.cd = cd  # 阻力系数

        self.msisDragEff.atmoDensInMsg.subscribeTo(self.msisAtmo.envOutMsg)

        # 挂到飞行器并加入任务
        self.scObject.addDynamicEffector(self.msisDragEff)
        self.simulator.AddModelToTask(self.task_name, self.msisDragEff)


# 创建观测目标
n_targets = 3000  # 目标数量
n_ahead = 32 # 观测前瞻步数
# 观测包含“每个目标接下来 n_ahead 个机会窗口”的信息
#（如开始/结束时间、优先级、相对角度等），从而让 agent 能看到未来若干个时间步的机会分布，用于做规划决策（例如决定现在是充电还是拍照以把握未来的高优先机会）。
# 它决定了 Image 动作的选择维度大小。
# 机会窗口的时间长度长度由几何条件（轨道位置、最小仰角、遮挡等）决定，不是固定的常数（我还不确定）


class Density(obs.Observation):  # Density 类用于把“未来若干时间箱内的拍照机会优先级总和”作为观测向量返回，供智能体决策（例如衡量近期目标密度以决定拍照/充电）。
    def __init__(
        self,
        interval_duration=60 * 3,  # 每个时间箱的长度
        intervals=20, # 时间箱的数量
        norm=5, # 归一化因子
    ):
        self.satellite: "sats.AccessSatellite"
        super().__init__()
        self.interval_duration = interval_duration
        self.intervals = intervals
        self.norm = norm

    def get_obs(self):
        if self.intervals == 0:
            return []

        self.satellite.calculate_additional_windows(  # 更新卫星的可见窗口信息，直到指定的未来时间点
            self.simulator.sim_time
            + (self.intervals + 1) * self.interval_duration
            - self.satellite.window_calculation_time
        )
        soonest = self.satellite.upcoming_opportunities_dict(types="target") # 获取未来所有拍照机会中每个目标的最近一次机会
        rewards = np.array([opportunity.priority for opportunity in soonest])
        times = np.array([opportunities[0][1] for opportunities in soonest.values()])
        time_bins = np.floor((times - self.simulator.sim_time) / self.interval_duration)
        densities = [sum(rewards[time_bins == i]) for i in range(self.intervals)]
        return np.array(densities) / self.norm

def wheel_speed_3(sat):
    return np.array(sat.dynamics.wheel_speeds[0:3]) / 630

def s_hat_H(sat):  #计算从卫星指向太阳的单位向量，并将其表示在卫星的希尔坐标系（Hill Frame / Orbit Frame）下
    r_SN_N = (
        sat.simulator.world.gravFactory.spiceObject.planetStateOutMsgs[  # 从仿真环境的 SPICE 接口（行星星历库）中读取太阳相对于惯性原点（N）的位置向量。
            sat.simulator.world.sun_index  
        ]
        .read()
        .PositionVector
    )
    r_BN_N = sat.dynamics.r_BN_N 
    r_SB_N = np.array(r_SN_N) - np.array(r_BN_N)
    r_SB_H = rv2HN(r_BN_N, sat.dynamics.v_BN_N) @ r_SB_N
    return r_SB_H / np.linalg.norm(r_SB_H)    # 返回归一化后的单位向量

class ImagingSatellite(sats.ImagingSatellite):
    observation_spec = [  # 创建观测空间
        obs.SatProperties(
            dict(prop="omega_BH_H", norm=0.03),
            dict(prop="c_hat_H"),
            dict(prop="r_BN_P", norm=orbitalMotion.REQ_EARTH * 1e3),
            dict(prop="v_BN_P", norm=7616.5),
            dict(prop="battery_charge_fraction"),
            dict(prop="wheel_speed_3", fn=wheel_speed_3),
            dict(prop="s_hat_H", fn=s_hat_H),
        ),
        obs.OpportunityProperties(
            dict(prop="priority"),
            dict(prop="r_LB_H", norm=800 * 1e3),
            dict(prop="target_angle", norm=np.pi / 2),
            dict(prop="target_angle_rate", norm=0.03),
            dict(prop="opportunity_open", norm=300.0),
            dict(prop="opportunity_close", norm=300.0),
            type="target",
            n_ahead_observe=n_ahead,
        )
    ]
    action_spec = [act.Image(n_ahead_image=n_ahead),
                    act.Charge(duration=300),
                    act.Downlink(duration=300)]  # 定义动作空间，现在是拍照
    
    obs.Eclipse(norm=5700),  #日食
    Density(intervals=20, norm=5.0),  # 拍照的机会密度观测
    include_time = True
    if include_time == True:
        observation_spec.append(obs.Time())

    dyn_type = MSISDynModel  # 使用自定义的 MSIS 动力学模型 
    # dyn_type = dyn.FullFeaturedDynModel # 使用 bsk_rl 自带的 FullFeaturedDynModel 动力学模型
    fsw_type = fsw.SteeringImagerFSWModel


sat_args = dict(
    imageAttErrorRequirement=0.01,
    imageRateErrorRequirement=0.01,
    batteryStorageCapacity=80.0 * 3600 * 100.0,
    storedCharge_Init=80.0 * 3600 * 100.0,
    dataStorageCapacity=200 * 8e6 * 100,
    u_max=0.4,
    K1=0.25,
    K3=3.0,
    omega_max=0.087,
    servo_Ki=5.0,
    servo_P=30.0,
)

# 给卫星参数添加能量参数
sat_args_power={}
sat_args_power.update(sat_args)
sat_args_power.update(
    dict(
        batteryStorageCapacity=120.0 * 3600,
        storedCharge_Init=lambda: 120.0 * 3600 * np.random.uniform(0.4, 1.0),
        rwBasePower=20.4,
        instrumentPowerDraw=-10,
        thrusterPowerDraw=-30,
        nHat_B=np.array([0, 0, -1]),
        wheelSpeeds=lambda: np.random.uniform(-2000, 2000, 3),
        desatAttitude="nadir",
        # 启用轮速/力矩上限
        useMaxTorque=True,        # 允许轮子使用最大力矩限制（避免无限扭矩）
        rwMaxTorque=0.1,         # 轮子最大力矩（牛·米）
    )
)

# walker_delta_args 会生成一个函数，这个函数会在每次环境 reset 时被调用，以随机化卫星的初始轨道参数。
# walker_delta_args 是 bsk_rl.utils.orbital 提供的一个辅助函数，用来生成 Walker 星座风格的轨道元素（orbit elements）。
# 它返回一个“轨道生成器”（callable），按你给定的参数（如 altitude、inc、n_planes 等）计算每颗卫星的轨道偏移/相位，用于在创建多星时给每颗卫星分配不同的初始轨道（即实现 Walker δ 星座布局）。
sat_arg_randomizer = walker_delta_args(altitude=800.0, inc=60.0, n_planes=1) 

target_distribution = "uniform"  # 目标分布方式
if target_distribution == "uniform":
    targets = scene.UniformTargets(n_targets)  # 生成随意分布的目标
elif target_distribution == "cities":
    targets = scene.CityTargets(n_targets)  # 生成城市分布的目标

# 如何创建指定的地面站


def episode_data_callback(env):  # 在每个 episode 结束时收集数据
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
N_SATS = 8
TRAIN_ITERS = 200
TIME_LIMIT = 5700.0 * 2  # 每 episode 时长（秒）
N_CPUS = 8

training_args = dict(
    lr=0.00003,
    gamma=0.999,
    train_batch_size=2500,  # usually a larger number, like 2500
    num_sgd_iter=10,
    model=dict(fcnet_hiddens=[512, 512], vf_share_layers=False),
    lambda_=0.95,
    use_kl_loss=False,
    clip_param=0.1,
    grad_clip=0.5,
)

# 创建 env 构造器：每个 worker 在本地创建自己的 GeneralSatelliteTasking 实例
# 因为默认的SatelliteTasking-RLlib wrapper，它只接受单颗卫星
def make_leo_env_creator(n_sats, time_limit, sat_args, targets, sat_arg_randomizer):
    def _creator(env_config):
        satellites=[
        ImagingSatellite("EO-1", sat_args),
        ImagingSatellite("EO-2", sat_args),
        ImagingSatellite("EO-3", sat_args),
        ImagingSatellite("EO-4", sat_args),
        ImagingSatellite("EO-5", sat_args),
        ImagingSatellite("EO-6", sat_args),
        ImagingSatellite("EO-7", sat_args),
        ImagingSatellite("EO-8", sat_args),
        ImagingSatellite("EO-9", sat_args),
        ]

        # 自定义地面站配置：lat/long 为度，elev 为米
        gs_data = [
            dict(name="GS_Alaska",   lat=64.0,  long=-147.5, elev=0.0),
            dict(name="GS_Norway",   lat=69.0,  long=  18.9, elev=0.0),
            dict(name="GS_Australia",lat=-35.3, long= 149.1, elev=0.0),
        ]

        return GeneralSatelliteTasking(   # GeneralSatelliteTasking 支持多卫星协同任务
            satellites=satellites,
            scenario=targets,
            rewarder=data.UniqueImageReward(),  # 唯一图像奖励，此奖励需要与生成目标的场景（如UniformTargets或CityTargets）一起使用。
                                                # 还有一种叫ResourceReward，可以考虑能源等因素
            communicator=comm.LOSCommunication(),  # 卫星之间的视线通信，只有视线接触才可以通讯
            sat_arg_randomizer=sat_arg_randomizer,
            time_limit=time_limit,
            log_level="WARNING",  # 设置日志级别为 WARNING，避免过多输出，只显示警告和错误信息，INFO 级别会输出大量信息
            
            # 把 groundStationsData 传给 GroundStationWorldModel.setup_ground_locations
            world_args=dict(
                groundStationsData=gs_data,
                # 如需修改最小仰角/最大距离，也可以在这里加：
                gsMinimumElevation=np.radians(10.0),
                gsMaximumRange=-1,  # 负值表示无限制 Set to ``-1`` to disable.
            ),
            # 自己决定是否输出三维可视化，每一个episode都会生成一个文件，所以文件会很多
            vizard_dir="/workspace/learn_basilisk/vizard",
            vizard_settings=dict(showLocationLabels=0),  # 配置可视化动画
        )
    return _creator


# 注册环境
ENV_NAME = "LEO_multi_env"
register_env(ENV_NAME, make_leo_env_creator(N_SATS, TIME_LIMIT, sat_args, targets, sat_arg_randomizer))
ray.init(
    ignore_reinit_error=True,
    num_cpus=N_CPUS,
    object_store_memory=2_000_000_000  # 2 GB
)

config = (
    PPOConfig()
    .training(**training_args)
    .env_runners(num_env_runners=N_CPUS-1, sample_timeout_s=1000.0)
    .environment(
        env=ENV_NAME,      # 指定环境名称
        env_config={
        },  
    )
    .reporting(
        metrics_num_episodes_for_smoothing=1,   # 平滑奖励的 episode 数，n个 episode 求平均
        metrics_episode_collection_timeout_s=180, # 收集 episode 数据的超时时间，单位秒
    )
    .resources(num_gpus=1)       # 使用 1 个 GPU 进行训练
    .checkpointing(export_native_model_files=True) 
    .framework(framework="torch")
    .callbacks(WrappedEpisodeDataCallbacks)
)   

tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 20},  # Adjust the number of iterations as needed
    checkpoint_freq=1,
    checkpoint_at_end=True,
    storage_path="/workspace/learn_basilisk/ray_results",
)
# Shutdown Ray
ray.shutdown()

