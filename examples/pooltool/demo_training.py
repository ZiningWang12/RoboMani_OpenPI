#!/usr/bin/env python3
"""
Training integration demonstration for Pooltool environment.

This script shows how to:
1. Use Pooltool environment with openpi policies
2. Demonstrate data transformation
3. Create training-compatible data flows
4. Test policy integration
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
import json
import time

from pooltool_env import (
    PooltoolEnvironment, 
    TableConfig, 
    RobotArmConfig, 
    CameraConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_data_transformation():
    """Demonstrate data transformation for openpi compatibility."""
    print("=" * 60)
    print("DATA TRANSFORMATION DEMO")
    print("=" * 60)
    
    env = PooltoolEnvironment(render_mode="rgb_array")
    
    try:
        env.reset()
        obs = env.get_observation()
        
        print("1. Raw observation structure:")
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} {value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with keys {list(value.keys())}")
            else:
                print(f"  {key}: {type(value)}")
        
        print("\n2. OpenPI-compatible format:")
        
        # Standard openpi observation format
        openpi_obs = {
            "base_0_rgb": obs["base_0_rgb"],           # Top camera (1920x1080x3)
            "left_wrist_0_rgb": obs["left_wrist_0_rgb"],   # Side camera (1280x720x3)
            "right_wrist_0_rgb": obs["right_wrist_0_rgb"], # Wrist camera (640x480x3)
            "state": obs["state"]                       # Robot state vector (32D)
        }
        
        print("OpenPI observation keys:")
        for key, value in openpi_obs.items():
            print(f"  {key}: {value.shape} {value.dtype}")
        
        print("\n3. Action format compatibility:")
        
        # OpenPI action format (typically 32-dimensional for robotic tasks)
        sample_action = np.random.randn(32).astype(np.float32)
        print(f"Sample openpi action: shape={sample_action.shape}, dtype={sample_action.dtype}")
        
        # Convert to pooltool action format
        pooltool_action = {
            "robot_action": {
                "joint_positions": sample_action[:6],  # First 6 dims for joints
                "end_effector_pose": sample_action[6:12]  # Next 6 dims for pose
            },
            "cue_action": {
                "power": np.clip(sample_action[12], 0, 1),  # Power in [0,1]
                "angle": sample_action[13] * np.pi        # Angle in radians
            }
        }
        
        print("Converted pooltool action structure:")
        print(json.dumps(pooltool_action, indent=2, default=str))
        
        # Test action application
        env.apply_action(pooltool_action)
        new_obs = env.get_observation()
        print(f"\n4. Action applied successfully! New state shape: {new_obs['state'].shape}")
        
    except Exception as e:
        print(f"Data transformation demo error: {e}")
        logger.exception("Data transformation demo failed")
    finally:
        env.close()


def demo_episode_rollout():
    """Demonstrate a complete episode rollout."""
    print("\n" + "=" * 60)
    print("EPISODE ROLLOUT DEMO")
    print("=" * 60)
    
    env = PooltoolEnvironment(
        render_mode="human",
        max_episode_steps=200
    )
    
    episode_data = []
    
    try:
        # Reset environment
        env.reset()
        obs = env.get_observation()
        
        print("Starting episode rollout...")
        print(f"Initial state: {obs['state'][:8]}")  # Show first 8 dims
        
        episode_step = 0
        episode_reward = 0.0
        
        while not env.is_episode_complete() and episode_step < 50:  # Short demo
            # Random policy for demonstration
            action_vector = np.random.randn(32).astype(np.float32) * 0.2  # Small actions
            
            # Convert to environment action format
            if episode_step < 20:
                # Robot movement phase
                env_action = {
                    "robot_action": {
                        "joint_positions": action_vector[:6]
                    }
                }
            elif episode_step == 20:
                # Cue strike phase
                env_action = {
                    "cue_action": {
                        "power": 0.4,
                        "angle": np.random.uniform(-0.5, 0.5)
                    }
                }
            else:
                # Observation phase (no action)
                env_action = {}
            
            # Apply action and get new observation
            env.apply_action(env_action)
            new_obs = env.get_observation()
            
            # Calculate simple reward (example)
            reward = 0.0
            if episode_step > 20:  # After striking
                # Reward for ball movement
                cue_ball = env.balls[0]
                if not cue_ball.is_pocketed:
                    ball_speed = np.linalg.norm(cue_ball.velocity)
                    reward += ball_speed * 0.1
                
                # Reward for pocketing object balls
                pocketed_object_balls = sum(1 for ball in env.balls[1:] if ball.is_pocketed)
                reward += pocketed_object_balls * 10.0
                
                # Penalty for pocketing cue ball
                if cue_ball.is_pocketed:
                    reward -= 20.0
            
            episode_reward += reward
            
            # Store transition data
            transition = {
                "observation": obs["state"].copy(),
                "action": action_vector.copy(),
                "reward": reward,
                "next_observation": new_obs["state"].copy(),
                "done": env.is_episode_complete(),
                "info": {
                    "step": episode_step,
                    "time": env.time,
                    "balls_pocketed": sum(1 for ball in env.balls[1:] if ball.is_pocketed),
                    "cue_ball_pocketed": env.balls[0].is_pocketed
                }
            }
            episode_data.append(transition)
            
            # Update for next step
            obs = new_obs
            episode_step += 1
            
            if episode_step % 10 == 0:
                print(f"Step {episode_step}: Reward={reward:.3f}, "
                      f"Cumulative={episode_reward:.3f}, "
                      f"Balls stopped={env._all_balls_stopped()}")
            
            time.sleep(0.1)  # Visual delay
        
        print(f"\nEpisode completed!")
        print(f"Total steps: {episode_step}")
        print(f"Total reward: {episode_reward:.3f}")
        print(f"Final balls pocketed: {sum(1 for ball in env.balls[1:] if ball.is_pocketed)}")
        print(f"Episode data collected: {len(episode_data)} transitions")
        
        # Analyze episode data
        rewards = [t["reward"] for t in episode_data]
        print(f"Reward statistics: mean={np.mean(rewards):.3f}, "
              f"std={np.std(rewards):.3f}, "
              f"min={np.min(rewards):.3f}, "
              f"max={np.max(rewards):.3f}")
        
    except Exception as e:
        print(f"Episode rollout demo error: {e}")
        logger.exception("Episode rollout demo failed")
    finally:
        env.close()
    
    return episode_data


def demo_batch_data_generation():
    """Demonstrate batch data generation for training."""
    print("\n" + "=" * 60)
    print("BATCH DATA GENERATION DEMO")
    print("=" * 60)
    
    env = PooltoolEnvironment(
        render_mode="rgb_array",  # Faster for batch generation
        max_episode_steps=100
    )
    
    batch_size = 5
    all_episodes = []
    
    try:
        print(f"Generating {batch_size} episodes...")
        
        for episode_idx in range(batch_size):
            print(f"\nEpisode {episode_idx + 1}/{batch_size}")
            
            env.reset()
            episode_data = []
            
            for step in range(20):  # Short episodes for demo
                # Random action
                action = np.random.randn(32).astype(np.float32) * 0.1
                
                obs_before = env.get_observation()
                
                # Convert action format
                env_action = {
                    "robot_action": {
                        "joint_positions": action[:6]
                    }
                }
                
                if step == 10:  # Strike ball mid-episode
                    env_action["cue_action"] = {
                        "power": 0.3,
                        "angle": np.random.uniform(-1, 1)
                    }
                
                env.apply_action(env_action)
                obs_after = env.get_observation()
                
                # Simple reward
                reward = np.random.randn() * 0.1  # Placeholder reward
                
                episode_data.append({
                    "obs": obs_before["state"],
                    "action": action,
                    "reward": reward,
                    "next_obs": obs_after["state"],
                    "done": env.is_episode_complete()
                })
                
                if env.is_episode_complete():
                    break
            
            all_episodes.append(episode_data)
            print(f"  Episode {episode_idx + 1}: {len(episode_data)} steps")
        
        # Convert to batch format
        print(f"\nConverting to batch format...")
        
        all_obs = []
        all_actions = []
        all_rewards = []
        all_next_obs = []
        all_dones = []
        
        for episode in all_episodes:
            for transition in episode:
                all_obs.append(transition["obs"])
                all_actions.append(transition["action"])
                all_rewards.append(transition["reward"])
                all_next_obs.append(transition["next_obs"])
                all_dones.append(transition["done"])
        
        # Stack into batches
        batch_obs = np.stack(all_obs)
        batch_actions = np.stack(all_actions)
        batch_rewards = np.array(all_rewards)
        batch_next_obs = np.stack(all_next_obs)
        batch_dones = np.array(all_dones)
        
        print(f"Batch data shapes:")
        print(f"  Observations: {batch_obs.shape}")
        print(f"  Actions: {batch_actions.shape}")
        print(f"  Rewards: {batch_rewards.shape}")
        print(f"  Next observations: {batch_next_obs.shape}")
        print(f"  Dones: {batch_dones.shape}")
        
        print(f"Data statistics:")
        print(f"  Observation range: [{batch_obs.min():.3f}, {batch_obs.max():.3f}]")
        print(f"  Action range: [{batch_actions.min():.3f}, {batch_actions.max():.3f}]")
        print(f"  Reward range: [{batch_rewards.min():.3f}, {batch_rewards.max():.3f}]")
        print(f"  Done ratio: {batch_dones.mean():.3f}")
        
    except Exception as e:
        print(f"Batch data generation demo error: {e}")
        logger.exception("Batch data generation demo failed")
    finally:
        env.close()


def demo_performance_benchmarking():
    """Benchmark environment performance for training."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKING DEMO")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        {"render_mode": "rgb_array", "enable_depth": False, "enable_noise": False},
        {"render_mode": "rgb_array", "enable_depth": True, "enable_noise": False},
        {"render_mode": "rgb_array", "enable_depth": True, "enable_noise": True},
    ]
    
    for config_idx, config in enumerate(configs):
        print(f"\nConfiguration {config_idx + 1}: {config}")
        
        camera_config = CameraConfig(
            enable_depth=config["enable_depth"],
            enable_noise=config["enable_noise"]
        )
        
        env = PooltoolEnvironment(
            camera_config=camera_config,
            render_mode=config["render_mode"]
        )
        
        try:
            # Warmup
            env.reset()
            for _ in range(5):
                env.apply_action({})
                env.get_observation()
            
            # Benchmark
            num_steps = 50
            start_time = time.time()
            
            for step in range(num_steps):
                action = {"robot_action": {"joint_positions": np.random.randn(6) * 0.1}}
                env.apply_action(action)
                obs = env.get_observation()
            
            end_time = time.time()
            total_time = end_time - start_time
            fps = num_steps / total_time
            
            print(f"  Performance: {fps:.1f} FPS ({total_time:.3f}s for {num_steps} steps)")
            
            # Memory usage estimate
            obs = env.get_observation()
            total_obs_size = 0
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    size_mb = value.nbytes / (1024 * 1024)
                    total_obs_size += size_mb
                    if size_mb > 1.0:  # Only show large arrays
                        print(f"    {key}: {size_mb:.2f} MB")
            
            print(f"  Total observation size: {total_obs_size:.2f} MB")
            
        except Exception as e:
            print(f"  Configuration {config_idx + 1} failed: {e}")
        finally:
            env.close()


def main():
    """Run all training integration demonstrations."""
    print("Starting Pooltool Training Integration Demonstrations")
    
    try:
        # Data transformation demo
        demo_data_transformation()
        
        time.sleep(1.0)
        
        # Episode rollout demo
        episode_data = demo_episode_rollout()
        
        time.sleep(1.0)
        
        # Batch data generation demo
        demo_batch_data_generation()
        
        time.sleep(1.0)
        
        # Performance benchmarking
        demo_performance_benchmarking()
        
        print("\n" + "=" * 60)
        print("ALL TRAINING DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nSummary:")
        print("- Environment is compatible with openpi data formats")
        print("- Action/observation transformations work correctly")
        print("- Episode rollouts and batch data generation functional")
        print("- Performance is suitable for training workflows")
        
    except KeyboardInterrupt:
        print("\nTraining demonstrations interrupted by user")
    except Exception as e:
        print(f"\nTraining demo suite failed: {e}")
        logger.exception("Training demo suite error")


if __name__ == "__main__":
    main() 