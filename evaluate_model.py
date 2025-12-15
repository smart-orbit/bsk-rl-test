import os
os.environ["RAY_DEDUP_LOGS"] = "0"

import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.tune.registry import register_env
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 导入你的环境定义（从 density_reward_filter.py）
from density_reward_filter import (
    make_wrapped_env,
    MySatellite,
    StationKeepingReward,
    time_density,
    random_circular_orbit,
    orbitalMotion,
    _calculate_sma_basilisk,
    SMAFilter,  # ✅ 添加滤波器导入
)

# ========== 配置 ==========
CHECKPOINT_PATH = "/workspace/learn_basilisk/ray_results/PPO_2025-12-14_13-24-03/PPO_SatelliteTasking-Density_28886_00000_0_2025-12-14_13-24-03/checkpoint_000039"

NUM_EPISODES = 3
SAVE_PLOTS = True

# ========== 初始化 Ray（评估时用更少资源）==========
ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0)

# ========== 重新注册环境 ==========
register_env("SatelliteTasking-Density", make_wrapped_env)

# ========== 创建环境配置 ==========
oe = random_circular_orbit(alt=300.0, i=0.0, Omega=0.0, f=0.0)
sat_args = {"oe": oe}
sat = MySatellite(name="EO1", sat_args=sat_args)

gs_data = [
    dict(name="GS_Alaska",   lat=64.0,  long=-147.5, elev=0.0),
    dict(name="GS_Norway",   lat=69.0,  long=  18.9, elev=0.0),
    dict(name="GS_Australia",lat=-35.3, long= 149.1, elev=0.0),
]

my_rewarder = StationKeepingReward(target_alt_km=293.0, dist_penalty_scale=0.5, fuel_cost=3)

env_config = dict(
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
    density_schedule=time_density,
    thrust_gate_type="both",
    thrust_gate_angle_tol=5.0,
)

# ========== 从 checkpoint 加载模型（评估模式）==========
print(f"Loading model from: {CHECKPOINT_PATH}")

try:
    config = (
        PPOConfig()
        .environment(env="SatelliteTasking-Density", env_config=env_config)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=0)
        .evaluation(evaluation_num_env_runners=0)
        .training(
            model={"fcnet_hiddens": [512, 512], "vf_share_layers": False}
        )
    )
    
    algo = config.build()
    algo.restore(CHECKPOINT_PATH)
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
    ray.shutdown()
    exit(1)

# ========== 创建验证环境 ==========
env = make_wrapped_env(env_config)

# ========== 运行验证 ==========
for episode in range(NUM_EPISODES):
    print(f"\n{'='*50}")
    print(f"Episode {episode + 1}/{NUM_EPISODES}")
    print(f"{'='*50}")
    
    # ✅ 为每个episode创建独立的滤波器
    sma_filter = SMAFilter(method="moving_average", window_size=150)
    
    episode_data = {
        "sim_time": [],
        "altitude_km": [],
        "sma_km": [],
        "sma_km_filtered": [],  # ✅ 添加滤波后的半长轴
        "sma_deviation_km": [],
        "sma_deviation_km_filtered": [],  # ✅ 添加滤波后的偏差
        "reward": [],
        "dv_available": [],
        "density": [],
        "actions": [],
    }
    
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done:
        action = algo.compute_single_action(obs, explore=False)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if isinstance(reward, dict):
            reward_value = sum(reward.values()) / len(reward) if reward else 0.0
        else:
            reward_value = float(reward)
        
        total_reward += reward_value
        step_count += 1
        
        try:
            r_BN_N = np.array(env.unwrapped.satellite.dynamics.r_BN_N)
            v_BN_N = np.array(env.unwrapped.satellite.dynamics.v_BN_N)
            alt_km = np.linalg.norm(r_BN_N) / 1000.0 - orbitalMotion.REQ_EARTH
            
            # ✅ 计算原始半长轴
            sma_km = _calculate_sma_basilisk(r_BN_N, v_BN_N)
            # ✅ 计算滤波后的半长轴
            sma_km_filtered = sma_filter.update(sma_km)
            
            ref_sma_km = orbitalMotion.REQ_EARTH + 293.0
            sma_deviation_km = sma_km - ref_sma_km
            sma_deviation_km_filtered = sma_km_filtered - ref_sma_km
            
            dv_available = env.unwrapped.satellite.fsw.dv_available
            density = env.density_model.baseDensity
            sim_time = env._get_sim_time()
            
            episode_data["sim_time"].append(sim_time)
            episode_data["altitude_km"].append(alt_km)
            episode_data["sma_km"].append(sma_km)
            episode_data["sma_km_filtered"].append(sma_km_filtered)
            episode_data["sma_deviation_km"].append(sma_deviation_km)
            episode_data["sma_deviation_km_filtered"].append(sma_deviation_km_filtered)
            episode_data["reward"].append(reward_value)
            episode_data["dv_available"].append(dv_available)
            episode_data["density"].append(density)
            episode_data["actions"].append(action)
        except Exception as e:
            print(f"Error recording data: {e}")
        
        if step_count % 100 == 0:
            print(f"  Step {step_count}: reward={reward_value:.4f}, "
                  f"alt={alt_km:.2f} km, sma_dev={sma_deviation_km:.2f} km, "
                  f"sma_dev_filt={sma_deviation_km_filtered:.2f} km, "
                  f"dv={dv_available:.2f} m/s")
    
    print(f"\nEpisode {episode + 1} finished:")
    print(f"  Total steps: {step_count}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Average reward: {total_reward/step_count:.4f}")
    if len(episode_data['altitude_km']) > 0:
        print(f"  Final altitude: {episode_data['altitude_km'][-1]:.2f} km")
        print(f"  Final SMA (raw): {episode_data['sma_km'][-1]:.2f} km")
        print(f"  Final SMA (filtered): {episode_data['sma_km_filtered'][-1]:.2f} km")
        print(f"  Final SMA deviation (raw): {episode_data['sma_deviation_km'][-1]:.2f} km")
        print(f"  Final SMA deviation (filtered): {episode_data['sma_deviation_km_filtered'][-1]:.2f} km")
        print(f"  Fuel remaining: {episode_data['dv_available'][-1]:.2f} m/s")
        print(f"  Fuel consumed: {293.0 - episode_data['dv_available'][-1]:.2f} m/s")
    
    # ========== 保存图表（6个子图）==========
    if SAVE_PLOTS and len(episode_data["sim_time"]) > 0:
        fig, axes = plt.subplots(6, 1, figsize=(16, 24), sharex=True)
        
        # 1. Altitude
        axes[0].plot(episode_data["sim_time"], episode_data["altitude_km"], 'b-', linewidth=1)
        axes[0].axhline(y=293.0, color='r', linestyle='--', label='Target (293 km)')
        axes[0].set_ylabel("Altitude (km)")
        axes[0].set_title(f"Episode {episode + 1} - Satellite Altitude")
        axes[0].legend()
        axes[0].grid(True)
        
        # 2. SMA (Raw vs Filtered) ✅ 修改：同时显示原始和滤波
        axes[1].plot(episode_data["sim_time"], episode_data["sma_km"], 
                    'b-', alpha=0.3, linewidth=1, label='Raw SMA')
        axes[1].plot(episode_data["sim_time"], episode_data["sma_km_filtered"], 
                    'g-', linewidth=2, label='Filtered SMA')
        target_sma = orbitalMotion.REQ_EARTH + 293.0
        axes[1].axhline(y=target_sma, color='r', linestyle='--', label=f'Target ({target_sma:.1f} km)')
        axes[1].set_ylabel("Semi-Major Axis (km)")
        axes[1].set_title("Semi-Major Axis (Raw vs Filtered)")
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. SMA Deviation (Raw vs Filtered) ✅ 修改：同时显示原始和滤波偏差
        axes[2].plot(episode_data["sim_time"], episode_data["sma_deviation_km"], 
                    'b-', alpha=0.3, linewidth=1, label='Raw Deviation')
        axes[2].plot(episode_data["sim_time"], episode_data["sma_deviation_km_filtered"], 
                    'purple', linewidth=2, label='Filtered Deviation')
        axes[2].axhline(y=0, color='r', linestyle='--', label='Target (0 km)')
        axes[2].set_ylabel("SMA Deviation (km)")
        axes[2].set_title("Semi-Major Axis Deviation from Target (Raw vs Filtered)")
        axes[2].legend()
        axes[2].grid(True)
        
        # 4. Reward
        axes[3].plot(episode_data["sim_time"], episode_data["reward"], 'm-', linewidth=1)
        axes[3].set_ylabel("Step Reward")
        axes[3].set_title("Reward per Step")
        axes[3].grid(True)
        
        # 5. Fuel
        axes[4].plot(episode_data["sim_time"], episode_data["dv_available"], 'c-', linewidth=1)
        axes[4].set_ylabel("Delta-V Available (m/s)")
        axes[4].set_title("Remaining Fuel")
        axes[4].grid(True)
        
        # 6. Density
        axes[5].plot(episode_data["sim_time"], episode_data["density"], 'orange', linewidth=1)
        axes[5].set_ylabel("Density (kg/m³)")
        axes[5].set_xlabel("Simulation Time (s)")
        axes[5].set_title("Atmospheric Density")
        axes[5].grid(True)
        
        plt.tight_layout()
        plot_path = f"/workspace/learn_basilisk/logs/eval_episode_{episode + 1}.png"
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Plot saved to: {plot_path}")

# ========== 清理 ==========
env.close()
algo.stop()
ray.shutdown()
print("\nEvaluation complete!")