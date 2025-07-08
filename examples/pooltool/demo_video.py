#!/usr/bin/env python3
"""
Video visualization demo for Pooltool environment.

This script demonstrates:
1. Recording environment videos during gameplay
2. Creating animated visualizations
3. Saving video files for sharing and analysis
4. Multi-angle video capture
"""

import numpy as np
import time
import logging
import os
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("OpenCV not available. Install with: pip install opencv-python")
    CV2_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not available. Install with: pip install matplotlib")
    MATPLOTLIB_AVAILABLE = False

from pooltool_env import (
    PooltoolEnvironment, 
    TableConfig, 
    RobotArmConfig, 
    CameraConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoRecorder:
    """Video recording utility for pooltool environment."""
    
    def __init__(self, output_dir: str = "videos", fps: int = 30):
        self.output_dir = output_dir
        self.fps = fps
        self.writers = {}
        self.frames = {}
        self.recording = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def start_recording(self, video_name: str, camera_views: List[str]):
        """Start recording multiple camera views."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - cannot record video")
            return
        
        self.video_name = video_name
        self.recording = True
        
        # Initialize video writers and frame buffers
        for view in camera_views:
            self.frames[view] = []
            self.writers[view] = None
    
    def add_frame(self, view: str, frame: np.ndarray):
        """Add a frame to the recording."""
        if not self.recording or not CV2_AVAILABLE:
            return
        
        if view in self.frames:
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            self.frames[view].append(frame_bgr)
    
    def stop_recording(self):
        """Stop recording and save videos."""
        if not self.recording or not CV2_AVAILABLE:
            return
        
        self.recording = False
        
        for view, frames in self.frames.items():
            if len(frames) == 0:
                continue
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(self.output_dir, f"{self.video_name}_{view}.mp4")
            
            writer = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))
            
            # Write all frames
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            logger.info(f"Saved video: {video_path} ({len(frames)} frames)")
        
        # Clear frames
        self.frames.clear()


def demo_basic_gameplay_video():
    """Record a basic gameplay session with multiple camera angles."""
    print("=" * 60)
    print("BASIC GAMEPLAY VIDEO RECORDING")
    print("=" * 60)
    
    # Create video recorder
    recorder = VideoRecorder(output_dir="videos", fps=30)
    
    # Create environment
    env = PooltoolEnvironment(
        render_mode="rgb_array",  # Use rgb_array for video recording
        num_balls=9,              # 9-ball for shorter demo
        max_episode_steps=500
    )
    
    try:
        env.reset()
        
        # Start recording all camera views
        camera_views = ["top", "side", "wrist"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder.start_recording(f"basic_gameplay_{timestamp}", camera_views)
        
        print("Starting basic gameplay recording...")
        print("Recording 3 camera angles: top, side, wrist")
        
        step_count = 0
        phase = "positioning"  # positioning -> aiming -> shooting -> watching
        
        while step_count < 200:  # About 6-7 seconds at 30fps
            obs = env.get_observation()
            
            # Record frames from all cameras
            recorder.add_frame("top", obs["base_0_rgb"])
            recorder.add_frame("side", obs["left_wrist_0_rgb"])
            recorder.add_frame("wrist", obs["right_wrist_0_rgb"])
            
            # Execute different phases of gameplay
            if phase == "positioning" and step_count < 60:
                # Position robot arm
                target_joints = np.array([0.3, -0.2, 0.1, 0.0, 0.2, 0.0])
                action = {
                    "robot_action": {
                        "joint_positions": target_joints
                    }
                }
                
            elif phase == "positioning" and step_count >= 60:
                phase = "aiming"
                print(f"  Step {step_count}: Switching to aiming phase")
                
            elif phase == "aiming" and step_count < 90:
                # Fine-tune position and aim
                cue_ball_pos = env.balls[0].position
                aim_pos = np.array([
                    cue_ball_pos[0] - 0.1,
                    cue_ball_pos[1],
                    env.table_config.table_height + 0.05
                ])
                
                action = {
                    "robot_action": {
                        "end_effector_pose": np.concatenate([
                            aim_pos,
                            [0.0, -0.3, 0.0]  # Angle down toward cue ball
                        ])
                    }
                }
                
            elif phase == "aiming" and step_count >= 90:
                phase = "shooting"
                print(f"  Step {step_count}: Taking the shot!")
                
            elif phase == "shooting" and step_count == 90:
                # Take the shot
                action = {
                    "cue_action": {
                        "power": 0.4,
                        "angle": 0.1
                    }
                }
                
            elif phase == "shooting" and step_count > 90:
                phase = "watching"
                print(f"  Step {step_count}: Watching ball movement...")
                
            else:  # watching phase
                # No action, just observe
                action = {}
            
            # Apply action and update environment
            env.apply_action(action)
            
            # Progress indicator
            if step_count % 30 == 0:
                print(f"  Recording progress: {step_count}/200 frames ({step_count/200*100:.1f}%)")
            
            step_count += 1
            time.sleep(0.033)  # ~30 FPS timing
        
        # Stop recording
        recorder.stop_recording()
        
        print("\n✅ Video recording completed!")
        print(f"Videos saved in: {recorder.output_dir}/")
        print("Files created:")
        for view in camera_views:
            video_file = f"basic_gameplay_{timestamp}_{view}.mp4"
            print(f"  - {video_file}")
        
    except Exception as e:
        print(f"Video recording error: {e}")
        logger.exception("Video recording failed")
    finally:
        env.close()


def demo_strategic_gameplay_video():
    """Record a strategic gameplay session with AI planning visualization."""
    print("\n" + "=" * 60)
    print("STRATEGIC GAMEPLAY VIDEO RECORDING")
    print("=" * 60)
    
    if not CV2_AVAILABLE:
        print("OpenCV not available - skipping video recording")
        return
    
    # Create enhanced video recorder
    recorder = VideoRecorder(output_dir="videos", fps=24)  # Cinematic 24fps
    
    # Create environment with enhanced visualization
    camera_config = CameraConfig(
        top_camera_resolution=(1920, 1080),     # Full HD
        side_camera_resolution=(1280, 720),     # HD
        wrist_camera_resolution=(640, 480),     # VGA
        enable_noise=False,                     # Clean video
        lighting_brightness=1.2,                # Enhanced lighting
        render_quality="high"                   # Best quality
    )
    
    env = PooltoolEnvironment(
        camera_config=camera_config,
        render_mode="rgb_array",
        num_balls=16,  # Full rack
        max_episode_steps=1000
    )
    
    try:
        env.reset()
        
        # Start recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder.start_recording(f"strategic_gameplay_{timestamp}", ["top", "side", "wrist"])
        
        print("Recording strategic gameplay with AI planning...")
        print("This will demonstrate:")
        print("- Ball analysis and target selection")
        print("- Precise robot positioning")
        print("- Strategic shot execution")
        print("- Multi-ball sequences")
        
        step_count = 0
        shots_taken = 0
        max_shots = 3
        
        while step_count < 400 and shots_taken < max_shots:
            obs = env.get_observation()
            
            # Record all camera angles
            recorder.add_frame("top", obs["base_0_rgb"])
            recorder.add_frame("side", obs["left_wrist_0_rgb"])
            recorder.add_frame("wrist", obs["right_wrist_0_rgb"])
            
            # Strategic gameplay phases
            if step_count % 120 == 0 and shots_taken < max_shots:
                # Take a shot every 120 frames (5 seconds)
                shot_number = shots_taken + 1
                print(f"  Shot {shot_number}: Strategic planning...")
                
                # Position for shot
                target_angle = np.random.uniform(-0.5, 0.5)
                power = 0.3 + 0.2 * shots_taken  # Increasing power
                
                # Robot positioning
                cue_ball = env.balls[0]
                robot_pos = np.array([
                    cue_ball.position[0] - 0.15 * np.cos(target_angle),
                    cue_ball.position[1] - 0.15 * np.sin(target_angle),
                    env.table_config.table_height + 0.1
                ])
                
                action = {
                    "robot_action": {
                        "end_effector_pose": np.concatenate([
                            robot_pos,
                            [0.0, target_angle, 0.0]
                        ])
                    }
                }
                
            elif (step_count % 120) == 20 and step_count > 0:
                # Execute shot 20 frames after positioning
                shot_power = 0.3 + 0.15 * (shots_taken % 3)
                shot_angle = np.random.uniform(-0.3, 0.3)
                
                action = {
                    "cue_action": {
                        "power": shot_power,
                        "angle": shot_angle
                    }
                }
                
                shots_taken += 1
                print(f"    Executing shot {shots_taken} (power: {shot_power:.2f}, angle: {shot_angle:.2f})")
                
            else:
                # Observation and minor adjustments
                action = {}
            
            env.apply_action(action)
            
            # Progress and analysis
            if step_count % 60 == 0:
                balls_remaining = sum(1 for ball in env.balls[1:] if not ball.is_pocketed)
                balls_pocketed = len(env.balls) - 1 - balls_remaining
                print(f"  Frame {step_count}: {balls_pocketed} balls pocketed, {balls_remaining} remaining")
            
            step_count += 1
            time.sleep(0.042)  # 24 FPS timing
        
        recorder.stop_recording()
        
        # Final statistics
        final_pocketed = sum(1 for ball in env.balls[1:] if ball.is_pocketed)
        print(f"\n✅ Strategic gameplay video completed!")
        print(f"Final score: {final_pocketed} balls pocketed in {shots_taken} shots")
        print(f"Success rate: {final_pocketed/shots_taken*100:.1f}% balls per shot" if shots_taken > 0 else "No shots taken")
        
    except Exception as e:
        print(f"Strategic video recording error: {e}")
        logger.exception("Strategic video recording failed")
    finally:
        env.close()


def demo_robot_arm_showcase_video():
    """Create a showcase video focusing on robot arm capabilities."""
    print("\n" + "=" * 60)
    print("ROBOT ARM SHOWCASE VIDEO")
    print("=" * 60)
    
    if not CV2_AVAILABLE:
        print("OpenCV not available - skipping video recording")
        return
    
    recorder = VideoRecorder(output_dir="videos", fps=30)
    
    # High-quality configuration for showcase
    robot_config = RobotArmConfig(
        base_position=(0.0, 0.635, 0.79),
        max_reach=1.4,
        end_effector_precision=0.001,  # High precision
        max_velocity=0.8  # Smooth movements
    )
    
    camera_config = CameraConfig(
        top_camera_resolution=(1920, 1080),
        side_camera_resolution=(1280, 720),
        wrist_camera_resolution=(640, 480),
        enable_noise=False,
        lighting_brightness=1.3,
        render_quality="high"
    )
    
    env = PooltoolEnvironment(
        robot_config=robot_config,
        camera_config=camera_config,
        render_mode="rgb_array",
        num_balls=4  # Minimal balls for clean showcase
    )
    
    try:
        env.reset()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder.start_recording(f"robot_showcase_{timestamp}", ["top", "side", "wrist"])
        
        print("Recording robot arm capability showcase...")
        print("Demonstrating:")
        print("- Precise positioning and movement")
        print("- Smooth trajectory execution")
        print("- Different joint configurations")
        print("- Workspace coverage")
        
        # Define showcase sequence
        showcase_positions = [
            # (joint_angles, description, duration_frames)
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Home position", 60),
            (np.array([0.5, -0.3, 0.2, 0.1, -0.1, 0.0]), "Reach position 1", 90),
            (np.array([-0.4, 0.2, -0.1, 0.3, 0.2, 0.5]), "Reach position 2", 90),
            (np.array([0.0, -0.5, 0.4, -0.2, 0.1, -0.3]), "Reach position 3", 90),
            (np.array([0.6, 0.1, -0.3, 0.4, -0.2, 0.1]), "Reach position 4", 90),
            (np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "Return home", 60),
        ]
        
        step_count = 0
        current_sequence = 0
        sequence_start = 0
        
        while current_sequence < len(showcase_positions):
            obs = env.get_observation()
            
            # Record frames
            recorder.add_frame("top", obs["base_0_rgb"])
            recorder.add_frame("side", obs["left_wrist_0_rgb"])
            recorder.add_frame("wrist", obs["right_wrist_0_rgb"])
            
            # Get current showcase position
            target_joints, description, duration = showcase_positions[current_sequence]
            sequence_progress = step_count - sequence_start
            
            # Smooth interpolation to target position
            if sequence_progress == 0:
                print(f"  Sequence {current_sequence + 1}: {description}")
                start_joints = env.robot_arm.joint_angles.copy()
            
            # Linear interpolation with smoothing
            t = min(sequence_progress / duration, 1.0)
            # Smooth interpolation with easing
            t_smooth = 0.5 * (1 - np.cos(np.pi * t))  # Cosine easing
            
            current_joints = start_joints + t_smooth * (target_joints - start_joints)
            
            action = {
                "robot_action": {
                    "joint_positions": current_joints
                }
            }
            
            env.apply_action(action)
            
            # Check if sequence is complete
            if sequence_progress >= duration:
                current_sequence += 1
                sequence_start = step_count + 1
                if current_sequence < len(showcase_positions):
                    start_joints = env.robot_arm.joint_angles.copy()
            
            step_count += 1
            time.sleep(0.033)  # 30 FPS
        
        recorder.stop_recording()
        
        print(f"\n✅ Robot arm showcase video completed!")
        print(f"Demonstrated {len(showcase_positions)} different positions")
        print(f"Total frames recorded: {step_count}")
        
    except Exception as e:
        print(f"Robot showcase video recording error: {e}")
        logger.exception("Robot showcase video failed")
    finally:
        env.close()


def create_compilation_video():
    """Create a compilation video from all recorded videos."""
    print("\n" + "=" * 60)
    print("CREATING COMPILATION VIDEO")
    print("=" * 60)
    
    if not CV2_AVAILABLE:
        print("OpenCV not available - cannot create compilation")
        return
    
    video_dir = "videos"
    if not os.path.exists(video_dir):
        print(f"Video directory {video_dir} not found")
        return
    
    # Find all video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    if not video_files:
        print("No video files found to compile")
        return
    
    print(f"Found {len(video_files)} video files:")
    for vf in video_files:
        print(f"  - {vf}")
    
    # Group videos by type and select top view for compilation
    top_videos = [f for f in video_files if 'top' in f]
    
    if not top_videos:
        print("No top-view videos found for compilation")
        return
    
    print(f"\nCreating compilation from {len(top_videos)} top-view videos...")
    
    try:
        # Read first video to get dimensions
        first_video_path = os.path.join(video_dir, top_videos[0])
        cap = cv2.VideoCapture(first_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        
        # Create compilation video writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compilation_path = os.path.join(video_dir, f"pooltool_compilation_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(compilation_path, fourcc, fps, (width, height))
        
        # Add each video to compilation
        for video_file in sorted(top_videos):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            print(f"  Adding: {video_file}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            
            cap.release()
        
        writer.release()
        
        print(f"\n✅ Compilation video created: {compilation_path}")
        
    except Exception as e:
        print(f"Compilation creation error: {e}")
        logger.exception("Compilation creation failed")


def main():
    """Run all video demonstrations."""
    print("🎬 Starting Pooltool Video Demonstration Suite")
    print("=" * 60)
    
    if not CV2_AVAILABLE:
        print("⚠️  OpenCV is required for video recording")
        print("Install with: pip install opencv-python")
        return
    
    print("This will create several demonstration videos:")
    print("1. Basic gameplay with multiple camera angles")
    print("2. Strategic gameplay with AI planning")
    print("3. Robot arm capabilities showcase")
    print("4. Compilation video")
    print()
    
    # Check if videos directory exists, create if not
    os.makedirs("videos", exist_ok=True)
    
    try:
        # Run all video demos
        demo_basic_gameplay_video()
        
        time.sleep(2.0)
        
        demo_strategic_gameplay_video()
        
        time.sleep(2.0)
        
        demo_robot_arm_showcase_video()
        
        time.sleep(1.0)
        
        # Create compilation
        create_compilation_video()
        
        print("\n" + "=" * 60)
        print("🎉 ALL VIDEO DEMONSTRATIONS COMPLETED!")
        print("=" * 60)
        
        # List all created videos
        video_dir = "videos"
        if os.path.exists(video_dir):
            video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
            
            print(f"\nVideos created ({len(video_files)} files):")
            total_size = 0
            for vf in sorted(video_files):
                file_path = os.path.join(video_dir, vf)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                total_size += file_size
                print(f"  📹 {vf} ({file_size:.1f} MB)")
            
            print(f"\nTotal size: {total_size:.1f} MB")
            print(f"Videos location: {os.path.abspath(video_dir)}/")
            
            print("\n🎯 Video Types Created:")
            print("  • Multi-angle gameplay recordings")
            print("  • Strategic AI planning demonstrations")
            print("  • Robot arm capability showcases")
            print("  • Compilation video with highlights")
            
            print("\n💡 Usage Tips:")
            print("  • Videos are saved as MP4 format")
            print("  • Multiple camera angles provide different perspectives")
            print("  • Compilation video shows the best highlights")
            print("  • Videos can be shared or used for presentations")
        
    except KeyboardInterrupt:
        print("\n🛑 Video demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Video demonstration suite failed: {e}")
        logger.exception("Video demo suite error")


if __name__ == "__main__":
    main() 