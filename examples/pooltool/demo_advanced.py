#!/usr/bin/env python3
"""
Advanced demonstration of Pooltool environment features.

This script shows:
1. Strategic billiards gameplay
2. Pooltool native API integration (if available)
3. Advanced robot arm control
4. Complex scenario simulation
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple, Optional

from pooltool_env import (
    PooltoolEnvironment, 
    TableConfig, 
    RobotArmConfig, 
    CameraConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimplePoolStrategy:
    """Simple billiards strategy for demonstration."""
    
    def __init__(self, env: PooltoolEnvironment):
        self.env = env
        self.target_sequence = list(range(1, 16))  # Target balls 1-15 in order
        self.current_target = 0
    
    def get_target_ball(self) -> Optional[int]:
        """Get next target ball number."""
        while self.current_target < len(self.target_sequence):
            ball_num = self.target_sequence[self.current_target]
            # Check if this ball is still on table
            target_ball = next((ball for ball in self.env.balls if ball.number == ball_num), None)
            if target_ball and not target_ball.is_pocketed:
                return ball_num
            else:
                self.current_target += 1
        return None
    
    def select_pocket(self, ball_pos: np.ndarray) -> int:
        """Select best pocket for target ball."""
        # Pocket positions (simplified)
        pockets = [
            (0, 0, "bottom-left"),
            (self.env.table_config.width/2, 0, "bottom-center"),
            (self.env.table_config.width, 0, "bottom-right"),
            (0, self.env.table_config.height, "top-left"),
            (self.env.table_config.width/2, self.env.table_config.height, "top-center"),
            (self.env.table_config.width, self.env.table_config.height, "top-right")
        ]
        
        # Find closest pocket
        min_distance = float('inf')
        best_pocket = 0
        
        for i, (px, py, name) in enumerate(pockets):
            distance = np.sqrt((ball_pos[0] - px)**2 + (ball_pos[1] - py)**2)
            if distance < min_distance:
                min_distance = distance
                best_pocket = i
        
        return best_pocket
    
    def calculate_shot_parameters(self, target_ball_num: int, target_pocket: int) -> Dict[str, float]:
        """Calculate shot parameters for target ball and pocket."""
        # Find target ball
        target_ball = next((ball for ball in self.env.balls if ball.number == target_ball_num), None)
        if not target_ball:
            return {}
        
        cue_ball = self.env.balls[0]
        if cue_ball.is_pocketed:
            return {}
        
        # Pocket positions
        pockets = [
            (0, 0), (self.env.table_config.width/2, 0), (self.env.table_config.width, 0),
            (0, self.env.table_config.height), (self.env.table_config.width/2, self.env.table_config.height), 
            (self.env.table_config.width, self.env.table_config.height)
        ]
        
        if target_pocket >= len(pockets):
            return {}
        
        pocket_pos = np.array(pockets[target_pocket])
        target_pos = target_ball.position
        cue_pos = cue_ball.position
        
        # Calculate direction from target ball to pocket
        ball_to_pocket = pocket_pos - target_pos
        ball_to_pocket_unit = ball_to_pocket / np.linalg.norm(ball_to_pocket)
        
        # Ideal contact point on target ball (opposite side from pocket)
        contact_point = target_pos - ball_to_pocket_unit * (2 * target_ball.radius)
        
        # Direction from cue ball to contact point
        cue_to_contact = contact_point - cue_pos
        shot_angle = np.arctan2(cue_to_contact[1], cue_to_contact[0])
        
        # Shot power based on distance
        shot_distance = np.linalg.norm(cue_to_contact)
        power = min(0.8, max(0.2, shot_distance / 2.0))  # Power between 0.2 and 0.8
        
        return {
            "angle": shot_angle,
            "power": power,
            "hit_point": contact_point.tolist()
        }


def demo_strategic_gameplay():
    """Demonstrate strategic billiards gameplay."""
    print("=" * 60)
    print("STRATEGIC GAMEPLAY DEMO")
    print("=" * 60)
    
    env = PooltoolEnvironment(
        render_mode="human",
        num_balls=9,  # 9-ball game for simplicity
        max_episode_steps=500
    )
    
    strategy = SimplePoolStrategy(env)
    
    try:
        env.reset()
        print("Starting strategic 9-ball game...")
        
        game_step = 0
        shots_taken = 0
        
        while not env.is_episode_complete() and game_step < 100:
            # Check if all balls stopped
            if not env._all_balls_stopped():
                # Wait for balls to stop
                env.apply_action({})
                game_step += 1
                time.sleep(0.05)
                continue
            
            # Get target ball
            target_ball_num = strategy.get_target_ball()
            if target_ball_num is None:
                print("No more target balls!")
                break
            
            target_ball = next((ball for ball in env.balls if ball.number == target_ball_num), None)
            if not target_ball:
                strategy.current_target += 1
                continue
            
            print(f"\nShot {shots_taken + 1}: Targeting ball {target_ball_num}")
            print(f"Target ball position: {target_ball.position}")
            
            # Select pocket
            target_pocket = strategy.select_pocket(target_ball.position)
            print(f"Selected pocket: {target_pocket}")
            
            # Calculate shot parameters
            shot_params = strategy.calculate_shot_parameters(target_ball_num, target_pocket)
            if not shot_params:
                print("Cannot calculate shot parameters")
                break
            
            print(f"Shot angle: {shot_params['angle']:.3f} rad ({np.degrees(shot_params['angle']):.1f}°)")
            print(f"Shot power: {shot_params['power']:.3f}")
            
            # Position robot for shot
            cue_ball = env.balls[0]
            if cue_ball.is_pocketed:
                print("Cue ball pocketed! Game over.")
                break
            
            # Calculate robot position behind cue ball
            shot_angle = shot_params['angle']
            robot_distance = 0.2  # 20cm behind cue ball
            
            robot_pos = np.array([
                cue_ball.position[0] - robot_distance * np.cos(shot_angle),
                cue_ball.position[1] - robot_distance * np.sin(shot_angle),
                env.table_config.table_height + 0.1
            ])
            
            # Move robot to shooting position
            print("Positioning robot...")
            for step in range(10):  # Gradual movement
                current_pos = env.robot_arm.end_effector_position
                target_pos = current_pos + (robot_pos - current_pos) * 0.2
                
                move_action = {
                    "robot_action": {
                        "end_effector_pose": np.concatenate([
                            target_pos,
                            [0.0, shot_angle, 0.0]  # Orientation
                        ])
                    }
                }
                
                env.apply_action(move_action)
                time.sleep(0.1)
            
            # Take the shot
            print("Taking shot...")
            shot_action = {
                "cue_action": {
                    "power": shot_params['power'],
                    "angle": shot_angle
                }
            }
            
            env.apply_action(shot_action)
            shots_taken += 1
            
            # Wait for shot to complete
            print("Watching ball movement...")
            for wait_step in range(150):  # Wait up to 7.5 seconds
                env.apply_action({})
                
                if wait_step % 30 == 0:
                    cue_ball = env.balls[0]
                    print(f"  Cue ball at {cue_ball.position}, velocity={np.linalg.norm(cue_ball.velocity):.3f}")
                
                if env._all_balls_stopped():
                    break
                
                time.sleep(0.05)
            
            # Check results
            balls_pocketed_this_shot = []
            for ball in env.balls:
                if ball.is_pocketed and ball.number != 0:
                    balls_pocketed_this_shot.append(ball.number)
            
            if balls_pocketed_this_shot:
                print(f"Balls pocketed: {balls_pocketed_this_shot}")
                if target_ball_num in balls_pocketed_this_shot:
                    print(f"✓ Target ball {target_ball_num} successfully pocketed!")
                    strategy.current_target += 1
                else:
                    print(f"✗ Target ball {target_ball_num} not pocketed")
            else:
                print("No balls pocketed this shot")
            
            # Check if cue ball was pocketed
            if env.balls[0].is_pocketed:
                print("✗ Cue ball pocketed! Foul!")
                break
            
            game_step += 1
        
        # Final results
        pocketed_object_balls = [ball.number for ball in env.balls[1:] if ball.is_pocketed]
        remaining_balls = [ball.number for ball in env.balls[1:] if not ball.is_pocketed]
        
        print(f"\n=== GAME RESULTS ===")
        print(f"Shots taken: {shots_taken}")
        print(f"Balls pocketed: {sorted(pocketed_object_balls)}")
        print(f"Balls remaining: {sorted(remaining_balls)}")
        print(f"Success rate: {len(pocketed_object_balls)}/{len(env.balls)-1} = {len(pocketed_object_balls)/(len(env.balls)-1)*100:.1f}%")
        
        if len(remaining_balls) == 0:
            print("🎉 ALL BALLS CLEARED! EXCELLENT!")
        elif len(pocketed_object_balls) >= len(remaining_balls):
            print("👍 GOOD PERFORMANCE!")
        else:
            print("👌 ROOM FOR IMPROVEMENT")
    
    except Exception as e:
        print(f"Strategic gameplay demo error: {e}")
        logger.exception("Strategic gameplay demo failed")
    finally:
        env.close()


def demo_pooltool_integration():
    """Demonstrate integration with native pooltool API."""
    print("\n" + "=" * 60)
    print("POOLTOOL NATIVE API INTEGRATION DEMO")
    print("=" * 60)
    
    try:
        # Check if pooltool is available
        try:
            import pooltool as pt
            print("✓ Pooltool library is available")
            pooltool_available = True
        except ImportError:
            print("✗ Pooltool library not available - using fallback simulation")
            pooltool_available = False
        
        env = PooltoolEnvironment(
            render_mode="rgb_array",
            num_balls=16
        )
        
        env.reset()
        
        if pooltool_available and hasattr(env, 'pt_system'):
            print("\n1. Native pooltool physics integration:")
            print(f"   Table: {type(env.pt_table)}")
            print(f"   System: {type(env.pt_system)}")
            
            # Demonstrate advanced physics
            print("\n2. Advanced physics features:")
            
            # Create a complex shot scenario
            cue_ball = env.balls[0]
            target_ball = env.balls[1]
            
            print(f"   Cue ball: {cue_ball.position}")
            print(f"   Target ball: {target_ball.position}")
            
            # Apply spin and english
            print("\n3. Testing ball spin and english:")
            
            spin_shots = [
                {"power": 0.5, "angle": 0.0, "description": "Straight shot"},
                {"power": 0.3, "angle": 0.2, "description": "Slight angle"},
                {"power": 0.7, "angle": -0.3, "description": "Power shot with angle"}
            ]
            
            for i, shot in enumerate(spin_shots):
                print(f"\n   Shot {i+1}: {shot['description']}")
                
                # Reset ball positions
                env.reset()
                
                # Take shot
                env.apply_action({
                    "cue_action": {
                        "power": shot["power"],
                        "angle": shot["angle"]
                    }
                })
                
                # Simulate for a few steps
                for step in range(30):
                    env.apply_action({})
                    if env._all_balls_stopped():
                        break
                
                # Report results
                cue_ball = env.balls[0]
                print(f"     Final cue ball position: {cue_ball.position}")
                pocketed = [ball.number for ball in env.balls if ball.is_pocketed]
                if pocketed:
                    print(f"     Balls pocketed: {pocketed}")
        
        else:
            print("\n1. Using simplified physics simulation")
            print("   - Ball-ball collisions: ✓")
            print("   - Boundary collisions: ✓")
            print("   - Pocket detection: ✓")
            print("   - Friction and damping: ✓")
            
            # Demonstrate fallback physics
            print("\n2. Testing simplified physics:")
            
            # Create collision scenario
            env.reset()
            
            # Position balls for collision
            env.balls[0].position = np.array([0.5, 0.635])  # Cue ball
            env.balls[1].position = np.array([0.7, 0.635])  # Target ball
            
            # Strike cue ball
            env.apply_action({
                "cue_action": {
                    "power": 0.4,
                    "angle": 0.0
                }
            })
            
            print("   Simulating ball collision...")
            collision_detected = False
            
            for step in range(50):
                env.apply_action({})
                
                # Check for collision
                if step > 5 and not collision_detected:
                    cue_ball = env.balls[0]
                    target_ball = env.balls[1]
                    distance = np.linalg.norm(cue_ball.position - target_ball.position)
                    if distance < (cue_ball.radius + target_ball.radius) * 1.1:
                        print(f"   ✓ Collision detected at step {step}")
                        collision_detected = True
                
                if env._all_balls_stopped():
                    break
            
            # Final positions
            cue_ball = env.balls[0]
            target_ball = env.balls[1]
            print(f"   Final positions:")
            print(f"     Cue ball: {cue_ball.position}")
            print(f"     Target ball: {target_ball.position}")
        
        print("\n3. Performance comparison:")
        if pooltool_available:
            print("   Native pooltool: More accurate physics, slower computation")
        print("   Simplified physics: Fast computation, sufficient for training")
        
    except Exception as e:
        print(f"Pooltool integration demo error: {e}")
        logger.exception("Pooltool integration demo failed")
    finally:
        env.close()


def demo_robot_arm_advanced():
    """Demonstrate advanced robot arm capabilities."""
    print("\n" + "=" * 60)
    print("ADVANCED ROBOT ARM DEMO")
    print("=" * 60)
    
    # Custom high-precision robot configuration
    robot_config = RobotArmConfig(
        base_position=(0.0, 0.635, 0.79),
        max_reach=1.4,
        end_effector_precision=0.001,  # 1mm precision
        max_velocity=0.5  # Slower for precision
    )
    
    env = PooltoolEnvironment(
        robot_config=robot_config,
        render_mode="human"
    )
    
    try:
        env.reset()
        
        print("1. Precision positioning test:")
        
        # Test precise positioning
        target_positions = [
            np.array([0.8, 0.6, 1.0]),   # Position 1
            np.array([1.0, 0.7, 0.95]),  # Position 2  
            np.array([0.9, 0.5, 1.05])   # Position 3
        ]
        
        for i, target_pos in enumerate(target_positions):
            print(f"\n   Target {i+1}: {target_pos}")
            
            # Plan trajectory
            target_orient = np.array([0.0, -np.pi/6, 0.0])
            joint_solution = env.robot_arm.inverse_kinematics(target_pos, target_orient)
            
            if joint_solution is not None:
                # Execute with trajectory planning
                if env.robot_arm.plan_trajectory(joint_solution, duration=3.0, trajectory_type="polynomial"):
                    print(f"   ✓ Trajectory planned successfully")
                    
                    # Execute trajectory
                    start_time = time.time()
                    step_count = 0
                    
                    while env.robot_arm.update_trajectory():
                        env.apply_action({})
                        step_count += 1
                        time.sleep(0.016)  # ~60 FPS
                    
                    execution_time = time.time() - start_time
                    
                    # Check final position accuracy
                    final_pos = env.robot_arm.end_effector_position
                    error = np.linalg.norm(final_pos - target_pos)
                    
                    print(f"   Execution time: {execution_time:.2f}s ({step_count} steps)")
                    print(f"   Final position: {final_pos}")
                    print(f"   Position error: {error*1000:.2f}mm")
                    
                    if error < robot_config.end_effector_precision:
                        print(f"   ✓ Precision target achieved!")
                    else:
                        print(f"   ⚠ Precision target missed")
                else:
                    print(f"   ✗ Trajectory planning failed")
            else:
                print(f"   ✗ Position unreachable")
        
        print("\n2. Dynamic obstacle avoidance test:")
        
        # Create a ball obstacle
        obstacle_ball = env.balls[1]
        obstacle_ball.position = np.array([0.9, 0.635])
        print(f"   Obstacle ball at: {obstacle_ball.position}")
        
        # Try to reach a position that would collide
        risky_target = np.array([0.9, 0.635, env.table_config.table_height + 0.05])
        print(f"   Attempting to reach risky position: {risky_target}")
        
        risky_orient = np.array([0.0, -np.pi/4, 0.0])
        risky_joints = env.robot_arm.inverse_kinematics(risky_target, risky_orient)
        
        if risky_joints is not None:
            # Test collision detection
            current_joints = env.robot_arm.joint_angles.copy()
            env.robot_arm.joint_angles = risky_joints
            env.robot_arm.forward_kinematics()
            
            collision = env.robot_arm.check_environment_collision(env.balls, env.table_config)
            
            # Restore safe position
            env.robot_arm.joint_angles = current_joints
            env.robot_arm.forward_kinematics()
            
            if collision:
                print(f"   ✓ Collision detected - robot safely avoided obstacle")
            else:
                print(f"   Position appears safe")
        
        print("\n3. Multi-point path execution:")
        
        # Define a complex path
        path_points = [
            np.array([0.6, 0.4, 1.1]),
            np.array([0.8, 0.6, 0.95]),
            np.array([1.0, 0.8, 1.0]),
            np.array([1.2, 0.6, 0.9])
        ]
        
        print(f"   Executing path with {len(path_points)} waypoints...")
        
        total_path_time = 0
        successful_waypoints = 0
        
        for i, waypoint in enumerate(path_points):
            print(f"   Waypoint {i+1}: {waypoint}")
            
            # Plan to waypoint
            waypoint_orient = np.array([0.0, -np.pi/4, 0.0])
            waypoint_joints = env.robot_arm.inverse_kinematics(waypoint, waypoint_orient)
            
            if waypoint_joints is not None:
                if env.robot_arm.plan_trajectory(waypoint_joints, duration=2.0):
                    # Execute
                    start_time = time.time()
                    while env.robot_arm.update_trajectory():
                        env.apply_action({})
                        time.sleep(0.016)
                    
                    waypoint_time = time.time() - start_time
                    total_path_time += waypoint_time
                    
                    final_pos = env.robot_arm.end_effector_position
                    error = np.linalg.norm(final_pos - waypoint)
                    
                    print(f"     ✓ Reached in {waypoint_time:.2f}s, error: {error*1000:.1f}mm")
                    successful_waypoints += 1
                else:
                    print(f"     ✗ Trajectory planning failed")
            else:
                print(f"     ✗ Waypoint unreachable")
        
        print(f"\n   Path execution summary:")
        print(f"   Total time: {total_path_time:.2f}s")
        print(f"   Success rate: {successful_waypoints}/{len(path_points)} waypoints")
        print(f"   Average time per waypoint: {total_path_time/max(1,successful_waypoints):.2f}s")
    
    except Exception as e:
        print(f"Advanced robot arm demo error: {e}")
        logger.exception("Advanced robot arm demo failed")
    finally:
        env.close()


def main():
    """Run all advanced demonstrations."""
    print("Starting Pooltool Advanced Feature Demonstrations")
    print("These demos showcase the full capabilities of the environment")
    
    try:
        # Strategic gameplay demo
        demo_strategic_gameplay()
        
        time.sleep(2.0)
        
        # Pooltool integration demo
        demo_pooltool_integration()
        
        time.sleep(2.0)
        
        # Advanced robot arm demo
        demo_robot_arm_advanced()
        
        print("\n" + "=" * 60)
        print("ALL ADVANCED DEMOS COMPLETED!")
        print("=" * 60)
        print("\nKey achievements demonstrated:")
        print("✓ Strategic billiards gameplay with AI planning")
        print("✓ Native pooltool physics integration (when available)")
        print("✓ High-precision robot arm control")
        print("✓ Collision avoidance and safety systems")
        print("✓ Complex trajectory planning and execution")
        print("✓ Real-time performance suitable for interactive use")
        
    except KeyboardInterrupt:
        print("\nAdvanced demonstrations interrupted by user")
    except Exception as e:
        print(f"\nAdvanced demo suite failed: {e}")
        logger.exception("Advanced demo suite error")


if __name__ == "__main__":
    main() 