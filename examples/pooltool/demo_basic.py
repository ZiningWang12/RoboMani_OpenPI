#!/usr/bin/env python3
"""
Basic demonstration of the Pooltool billiards environment.

This script shows how to:
1. Initialize the environment
2. Reset and get observations
3. Apply simple actions
4. Visualize the environment
"""

import numpy as np
import time
import logging
from typing import Dict, Any

from pooltool_env import (
    PooltoolEnvironment, 
    TableConfig, 
    RobotArmConfig, 
    CameraConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_environment():
    """Demonstrate basic environment functionality."""
    print("=" * 60)
    print("POOLTOOL ENVIRONMENT BASIC DEMO")
    print("=" * 60)
    
    # Create environment with default configuration
    env = PooltoolEnvironment(
        render_mode="human",  # For visual display
        num_balls=16,         # Full rack
        max_episode_steps=1000
    )
    
    try:
        # Reset environment
        print("\n1. Resetting environment...")
        env.reset()
        
        # Get initial observation
        obs = env.get_observation()
        print(f"Initial observation keys: {list(obs.keys())}")
        print(f"Number of balls: {len(env.balls)}")
        print(f"Robot arm position: {env.robot_arm.end_effector_position}")
        
        # Show environment state
        print(f"Cue ball position: {env.balls[0].position}")
        print(f"All balls stopped: {env._all_balls_stopped()}")
        
        # Demonstrate simple actions
        print("\n2. Demonstrating robot arm movement...")
        
        # Move robot arm to different positions
        for i in range(5):
            # Random joint positions within limits
            target_joints = np.random.uniform(-0.5, 0.5, 6)
            
            action = {
                "robot_action": {
                    "joint_positions": target_joints
                }
            }
            
            env.apply_action(action)
            time.sleep(0.5)  # Visual delay
            
            print(f"Step {i+1}: Joint angles = {env.robot_arm.joint_angles}")
        
        # Demonstrate cue ball strike
        print("\n3. Demonstrating cue ball strike...")
        
        # Position robot near cue ball
        cue_ball_pos = env.balls[0].position
        strike_pos = np.array([
            cue_ball_pos[0] - 0.15,  # 15cm behind ball
            cue_ball_pos[1],
            env.table_config.table_height + 0.05
        ])
        
        strike_action = {
            "robot_action": {
                "end_effector_pose": np.concatenate([
                    strike_pos, 
                    [0.0, 0.0, 0.0]  # Orientation
                ])
            }
        }
        
        env.apply_action(strike_action)
        time.sleep(1.0)
        
        # Strike the cue ball
        cue_action = {
            "cue_action": {
                "power": 0.3,  # Medium power
                "angle": 0.0   # Straight ahead
            }
        }
        
        env.apply_action(cue_action)
        print("Cue ball struck!")
        
        # Watch ball movement
        print("\n4. Watching ball physics...")
        for step in range(100):
            env.apply_action({})  # No action, just update physics
            
            if step % 20 == 0:
                cue_ball = env.balls[0]
                print(f"Step {step}: Cue ball at {cue_ball.position}, "
                      f"velocity = {np.linalg.norm(cue_ball.velocity):.3f}")
            
            if env._all_balls_stopped():
                print("All balls have stopped!")
                break
            
            time.sleep(0.05)
        
        # Final state
        obs_final = env.get_observation()
        print(f"\n5. Final state:")
        print(f"Episode complete: {env.is_episode_complete()}")
        print(f"Total steps: {env.step_count}")
        print(f"Simulation time: {env.time:.2f} seconds")
        
        # Show ball states
        pocketed_balls = [ball.number for ball in env.balls if ball.is_pocketed]
        if pocketed_balls:
            print(f"Pocketed balls: {pocketed_balls}")
        else:
            print("No balls pocketed")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        logger.exception("Demo failed")
    finally:
        env.close()
        print("\nDemo completed!")


def demo_camera_system():
    """Demonstrate multi-camera observation system."""
    print("\n" + "=" * 60)
    print("CAMERA SYSTEM DEMO")
    print("=" * 60)
    
    # Create environment with custom camera config
    camera_config = CameraConfig(
        enable_depth=True,
        enable_noise=True,
        noise_std=0.02,
        lighting_brightness=1.2
    )
    
    env = PooltoolEnvironment(
        camera_config=camera_config,
        render_mode="rgb_array"
    )
    
    try:
        env.reset()
        
        # Get observation with all camera views
        obs = env.get_observation()
        
        print("Camera observations:")
        print(f"- Top view: {obs['base_0_rgb'].shape}")
        print(f"- Side view: {obs['left_wrist_0_rgb'].shape}")
        print(f"- Wrist view: {obs['right_wrist_0_rgb'].shape}")
        
        if camera_config.enable_depth:
            top_depth = env.cameras["top"].get_depth_image()
            if top_depth is not None:
                print(f"- Top depth: {top_depth.shape}")
        
        # Test camera intrinsics
        for camera_type in ["top", "side", "wrist"]:
            intrinsics = camera_config.get_camera_intrinsics(camera_type)
            print(f"\n{camera_type.capitalize()} camera intrinsics:")
            print(f"  Focal length: ({intrinsics['fx']:.1f}, {intrinsics['fy']:.1f})")
            print(f"  Principal point: ({intrinsics['cx']:.1f}, {intrinsics['cy']:.1f})")
            print(f"  FOV: {intrinsics['fov']:.1f} degrees")
        
        # Move robot and observe wrist camera changes
        print(f"\nTesting wrist camera movement...")
        
        for i in range(3):
            # Random robot pose
            random_joints = np.random.uniform(-0.5, 0.5, 6)
            action = {"robot_action": {"joint_positions": random_joints}}
            
            env.apply_action(action)
            obs = env.get_observation()
            
            print(f"Step {i+1}: Robot at {env.robot_arm.end_effector_position}")
            print(f"  Wrist camera updated: {obs['right_wrist_0_rgb'].shape}")
        
    except Exception as e:
        print(f"Camera demo error: {e}")
        logger.exception("Camera demo failed")
    finally:
        env.close()


def demo_robot_arm_control():
    """Demonstrate advanced robot arm control features."""
    print("\n" + "=" * 60)
    print("ROBOT ARM CONTROL DEMO")
    print("=" * 60)
    
    # Custom robot configuration
    robot_config = RobotArmConfig(
        base_position=(0.0, 0.635, 0.79),
        max_reach=1.4,
        max_velocity=1.0
    )
    
    env = PooltoolEnvironment(
        robot_config=robot_config,
        render_mode="human"
    )
    
    try:
        env.reset()
        
        print("1. Testing forward kinematics...")
        for joint_set in [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Home position
            np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.0]),    # Random pose 1
            np.array([-0.3, 0.4, -0.2, 0.3, 0.2, 0.5])     # Random pose 2
        ]:
            env.robot_arm.joint_angles = joint_set
            pos, orient = env.robot_arm.forward_kinematics()
            print(f"  Joints {joint_set} -> Position {pos}, Orientation {orient}")
        
        print("\n2. Testing inverse kinematics...")
        target_positions = [
            np.array([0.8, 0.6, 1.0]),    # Reachable position
            np.array([1.2, 0.8, 0.9]),    # Another reachable position
            np.array([2.0, 2.0, 2.0])     # Unreachable position
        ]
        
        for target_pos in target_positions:
            target_orient = np.array([0.0, -np.pi/4, 0.0])
            joint_solution = env.robot_arm.inverse_kinematics(target_pos, target_orient)
            
            if joint_solution is not None:
                print(f"  Target {target_pos} -> Joints {joint_solution}")
                # Verify solution
                env.robot_arm.joint_angles = joint_solution
                achieved_pos, _ = env.robot_arm.forward_kinematics()
                error = np.linalg.norm(achieved_pos - target_pos)
                print(f"    Verification error: {error:.4f}m")
            else:
                print(f"  Target {target_pos} -> UNREACHABLE")
        
        print("\n3. Testing trajectory planning...")
        # Plan trajectory to a target
        target_joints = np.array([0.5, -0.5, 0.3, -0.2, 0.4, -0.1])
        
        if env.robot_arm.plan_trajectory(target_joints, duration=2.0):
            print(f"  Trajectory planned with {len(env.robot_arm.trajectory_buffer)} points")
            
            # Execute trajectory
            print("  Executing trajectory...")
            step = 0
            while env.robot_arm.update_trajectory():
                if step % 10 == 0:
                    pos = env.robot_arm.end_effector_position
                    print(f"    Step {step}: Position {pos}")
                step += 1
                time.sleep(0.02)
            
            print("  Trajectory completed!")
        else:
            print("  Trajectory planning failed")
        
        print("\n4. Testing collision detection...")
        # Test some collision scenarios
        test_cases = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),      # Safe position
            np.array([0.0, -1.5, 0.0, 0.0, 0.0, 0.0]),     # May cause table collision
        ]
        
        for joints in test_cases:
            env.robot_arm.joint_angles = joints
            env.robot_arm.forward_kinematics()
            
            env_collision = env.robot_arm.check_environment_collision(env.balls, env.table_config)
            self_collision = env.robot_arm.check_self_collision()
            
            print(f"  Joints {joints}:")
            print(f"    Environment collision: {env_collision}")
            print(f"    Self collision: {self_collision}")
        
    except Exception as e:
        print(f"Robot arm demo error: {e}")
        logger.exception("Robot arm demo failed")
    finally:
        env.close()


def main():
    """Run all demonstrations."""
    print("Starting Pooltool Environment Demonstrations")
    print("Press Ctrl+C to interrupt any demo")
    
    try:
        # Basic environment demo
        demo_basic_environment()
        
        time.sleep(1.0)
        
        # Camera system demo
        demo_camera_system()
        
        time.sleep(1.0)
        
        # Robot arm control demo
        demo_robot_arm_control()
        
        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nDemonstrations interrupted by user")
    except Exception as e:
        print(f"\nDemo suite failed: {e}")
        logger.exception("Demo suite error")


if __name__ == "__main__":
    main() 