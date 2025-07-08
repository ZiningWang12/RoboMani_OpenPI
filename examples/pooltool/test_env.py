"""
Test script for the Pooltool Environment
"""

import numpy as np
import time
from env import PooltoolEnvironment, make_example


def test_basic_environment():
    """Test basic environment functionality."""
    print("🎱 Testing Pooltool Environment...")
    
    # Create environment
    env = PooltoolEnvironment(
        table_size=(2.54, 1.27),
        num_balls=16,
        render_mode="rgb_array",
        seed=42
    )
    
    print(f"✅ Environment created with {len(env.balls)} balls")
    
    # Reset environment
    env.reset()
    print("✅ Environment reset successful")
    
    # Get initial observation
    obs = env.get_observation()
    print(f"✅ Initial observation keys: {list(obs.keys())}")
    print(f"   - Ball positions shape: {obs['ball_positions'].shape}")
    print(f"   - Ball velocities shape: {obs['ball_velocities'].shape}")
    print(f"   - Arm state shape: {obs['arm_state'].shape}")
    print(f"   - Image shape: {obs['image'].shape}")
    
    # Test arm movement
    print("\n🤖 Testing robot arm movement...")
    arm_action = {
        "arm_action": [0.5, 0.2, -0.1, 0.0, 0.0, 0.3]  # Joint angles
    }
    env.apply_action(arm_action)
    
    new_obs = env.get_observation()
    print(f"✅ Arm moved - new end effector position: {env.robot_arm.end_effector_pos}")
    
    # Test cue strike
    print("\n🎯 Testing cue ball strike...")
    cue_action = {
        "cue_action": {
            "power": 2.0,
            "angle": np.pi/4  # 45 degrees
        }
    }
    env.apply_action(cue_action)
    
    # Simulate a few steps
    print("\n⚡ Simulating physics for 5 seconds...")
    for i in range(300):  # 5 seconds at 60 FPS
        env.apply_action({})  # Empty action, just physics update
        
        if i % 60 == 0:  # Print every second
            cue_ball = env.balls[0]
            speed = np.linalg.norm(cue_ball.velocity)
            print(f"   Second {i//60}: Cue ball speed = {speed:.2f} m/s")
            
        if env.is_episode_complete():
            print(f"   Episode completed at step {i}")
            break
    
    # Final state
    final_obs = env.get_observation()
    final_positions = final_obs['ball_positions'].reshape(-1, 2)
    print(f"\n📊 Final ball positions:")
    for i, pos in enumerate(final_positions[:5]):  # Show first 5 balls
        print(f"   Ball {i}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    env.close()
    print("\n✅ Environment test completed successfully!")


def test_make_example():
    """Test the example data generation."""
    print("\n📝 Testing example data generation...")
    
    example = make_example()
    print(f"✅ Example generated with keys: {list(example.keys())}")
    print(f"   - Ball positions: {example['ball_positions'].shape}")
    print(f"   - Ball velocities: {example['ball_velocities'].shape}")
    print(f"   - Arm state: {example['arm_state'].shape}")
    print(f"   - Image: {example['image'].shape}")
    print(f"   - Prompt: '{example['prompt']}'")


def test_collision_physics():
    """Test collision physics."""
    print("\n💥 Testing collision physics...")
    
    # Create simple 2-ball scenario
    env = PooltoolEnvironment(num_balls=2, render_mode="rgb_array", seed=123)
    env.reset()
    
    # Position balls for collision
    env.balls[0].position = np.array([0.5, 0.635])  # Cue ball on left
    env.balls[1].position = np.array([1.0, 0.635])  # Target ball on right
    env.balls[0].velocity = np.array([2.0, 0.0])    # Moving right
    env.balls[1].velocity = np.array([0.0, 0.0])    # Stationary
    
    print(f"Before collision:")
    print(f"  Cue ball: pos={env.balls[0].position}, vel={env.balls[0].velocity}")
    print(f"  Target ball: pos={env.balls[1].position}, vel={env.balls[1].velocity}")
    
    # Simulate until collision and aftermath
    for i in range(100):
        env.apply_action({})
        
        # Check if collision occurred (balls moving apart)
        if i > 10 and np.linalg.norm(env.balls[1].velocity) > 0.1:
            print(f"\nAfter collision (step {i}):")
            print(f"  Cue ball: pos={env.balls[0].position}, vel={env.balls[0].velocity}")
            print(f"  Target ball: pos={env.balls[1].position}, vel={env.balls[1].velocity}")
            break
    
    env.close()
    print("✅ Collision physics test completed!")


if __name__ == "__main__":
    try:
        test_basic_environment()
        test_make_example()
        test_collision_physics()
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc() 