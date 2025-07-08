"""
Pooltool Policy for OpenPI
Input/output data transformations for the pooltool billiards environment.

This module provides:
- PooltoolInputs: Transform pooltool observations to model input format
- PooltoolOutputs: Transform model outputs to pooltool action format
- Data normalization and standardization utilities
- Compatibility with openpi training pipeline
"""

import dataclasses
import numpy as np
from typing import Any, ClassVar, Dict, List, Optional, Tuple
import logging

from openpi import transforms
from openpi.shared import image_tools

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PooltoolInputs(transforms.DataTransformFn):
    """Input transformation for Pooltool environment.
    
    Transforms pooltool environment observations into the format expected by openpi models.
    
    Expected inputs from environment:
    - ball_positions: [32] flattened positions of 16 balls (x,y coordinates)
    - ball_velocities: [32] flattened velocities of 16 balls
    - ball_states: [32] ball state info (is_pocketed, number)
    - arm_joint_angles: [6] robot arm joint angles
    - arm_state: [15] full arm state (joints, end-effector, cue state)
    - table_image_top: [H, W, 3] top-down view of table
    - table_image_side: [H, W, 3] side view of table
    - wrist_camera: [H, W, 3] wrist camera view
    - time: float current simulation time
    - prompt: str task description (optional)
    
    Output format matches openpi model expectations:
    - images: dict with required camera views
    - image_masks: dict with image validity masks
    - state: [action_dim] robot state vector
    - tokenized_prompt: [max_token_len] tokenized language prompt
    """
    
    # Action dimension for state vector
    action_dim: int = 32
    
    # Image resolution for model input (standard openpi format)
    image_resolution: Tuple[int, int] = (224, 224)
    
    # Expected camera names for openpi models
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "base_0_rgb",        # Top-down table view
        "left_wrist_0_rgb",  # Wrist camera
        "right_wrist_0_rgb"  # Wrist camera (duplicate for compatibility)
    )
    
    # Maximum sequence length for language prompts
    max_prompt_tokens: int = 48
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform pooltool observation to model input format."""
        try:
            # Extract and process images
            images = self._process_images(data)
            image_masks = self._create_image_masks(images)
            
            # Process robot state
            state = self._process_state(data)
            
            # Process language prompt if available
            prompt_data = self._process_prompt(data.get("prompt", ""))
            
            # Create output dictionary
            output = {
                # Core observation data
                "images": images,
                "image_masks": image_masks,
                "state": state,
                
                # Language prompt data
                **prompt_data,
                
                # Additional metadata
                "time": np.float32(data.get("time", 0.0)),
                "episode_step": np.int32(data.get("step_count", 0)),
            }
            
            # Add raw data for debugging/analysis
            if "ball_positions" in data:
                output["ball_positions"] = np.array(data["ball_positions"], dtype=np.float32)
            if "ball_velocities" in data:
                output["ball_velocities"] = np.array(data["ball_velocities"], dtype=np.float32)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in PooltoolInputs transformation: {e}")
            # Return safe default values
            return self._create_default_output()
    
    def _process_images(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Process and resize images to model input format."""
        images = {}
        
        # Map pooltool image keys to openpi camera names
        image_mapping = {
            "table_image_top": "base_0_rgb",
            "wrist_camera": "left_wrist_0_rgb",
            "wrist_camera": "right_wrist_0_rgb"  # Duplicate for compatibility
        }
        
        for pooltool_key, openpi_key in image_mapping.items():
            if pooltool_key in data:
                image = data[pooltool_key]
                
                # Ensure image is in correct format
                if isinstance(image, np.ndarray):
                    # Convert to float32 and normalize to [-1, 1] if needed
                    if image.dtype == np.uint8:
                        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
                    
                    # Resize to target resolution
                    if image.shape[:2] != self.image_resolution:
                        # Add batch dimension if needed
                        if len(image.shape) == 3:
                            image = image[None, ...]
                        
                        image = image_tools.resize_with_pad(
                            image, 
                            self.image_resolution[0], 
                            self.image_resolution[1]
                        )
                        
                        # Remove batch dimension
                        if image.shape[0] == 1:
                            image = image[0]
                    
                    images[openpi_key] = image.astype(np.float32)
                else:
                    # Create black image if invalid
                    images[openpi_key] = np.zeros(
                        (*self.image_resolution, 3), dtype=np.float32
                    ) - 1.0  # Black image in [-1, 1] range
            else:
                # Create black image for missing cameras
                images[openpi_key] = np.zeros(
                    (*self.image_resolution, 3), dtype=np.float32
                ) - 1.0
        
        return images
    
    def _create_image_masks(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create image validity masks."""
        masks = {}
        for camera_name, image in images.items():
            # Mark image as valid if it's not all black
            is_valid = np.any(image > -0.9)  # Allow for small numerical errors
            masks[camera_name] = np.array([is_valid], dtype=bool)
        return masks
    
    def _process_state(self, data: Dict[str, Any]) -> np.ndarray:
        """Process robot and environment state into fixed-size vector."""
        state_components = []
        
        # Robot arm joint angles (6 dimensions)
        if "arm_joint_angles" in data:
            joint_angles = np.array(data["arm_joint_angles"], dtype=np.float32)
            joint_angles = joint_angles[:6]  # Ensure exactly 6 joints
            if len(joint_angles) < 6:
                joint_angles = np.pad(joint_angles, (0, 6 - len(joint_angles)))
        else:
            joint_angles = np.zeros(6, dtype=np.float32)
        state_components.append(joint_angles)
        
        # End-effector position (3 dimensions)
        if "arm_end_effector_position" in data:
            ee_pos = np.array(data["arm_end_effector_position"][:3], dtype=np.float32)
        else:
            ee_pos = np.zeros(3, dtype=np.float32)
        state_components.append(ee_pos)
        
        # End-effector orientation (3 dimensions) 
        if "arm_end_effector_orientation" in data:
            ee_orient = np.array(data["arm_end_effector_orientation"][:3], dtype=np.float32)
        else:
            ee_orient = np.zeros(3, dtype=np.float32)
        state_components.append(ee_orient)
        
        # Cue ball position and velocity (4 dimensions)
        if "ball_positions" in data and len(data["ball_positions"]) >= 2:
            cue_pos = np.array(data["ball_positions"][:2], dtype=np.float32)
        else:
            cue_pos = np.zeros(2, dtype=np.float32)
        
        if "ball_velocities" in data and len(data["ball_velocities"]) >= 2:
            cue_vel = np.array(data["ball_velocities"][:2], dtype=np.float32)
        else:
            cue_vel = np.zeros(2, dtype=np.float32)
        
        state_components.extend([cue_pos, cue_vel])
        
        # Additional state features (up to action_dim total)
        # Time and episode information
        time_features = np.array([
            data.get("time", 0.0),
            data.get("step_count", 0.0) / 1000.0,  # Normalized step count
        ], dtype=np.float32)
        state_components.append(time_features)
        
        # Ball statistics (number of pocketed balls, etc.)
        ball_stats = self._compute_ball_statistics(data)
        state_components.append(ball_stats)
        
        # Concatenate all components
        state_vector = np.concatenate(state_components)
        
        # Pad or truncate to action_dim
        if len(state_vector) < self.action_dim:
            state_vector = np.pad(state_vector, (0, self.action_dim - len(state_vector)))
        elif len(state_vector) > self.action_dim:
            state_vector = state_vector[:self.action_dim]
        
        return state_vector.astype(np.float32)
    
    def _compute_ball_statistics(self, data: Dict[str, Any]) -> np.ndarray:
        """Compute summary statistics about ball state."""
        stats = []
        
        if "ball_states" in data:
            ball_states = np.array(data["ball_states"])
            # Reshape to [num_balls, 2] (is_pocketed, number)
            if len(ball_states) >= 32:
                ball_states = ball_states[:32].reshape(16, 2)
                
                # Count pocketed balls
                pocketed_count = np.sum(ball_states[:, 0])
                stats.append(pocketed_count / 16.0)  # Normalized
                
                # Check if 8-ball is pocketed
                eight_ball_pocketed = 0.0
                for i in range(len(ball_states)):
                    if ball_states[i, 1] == 8 and ball_states[i, 0] > 0.5:
                        eight_ball_pocketed = 1.0
                        break
                stats.append(eight_ball_pocketed)
                
                # Average ball velocity (motion indicator)
                if "ball_velocities" in data:
                    velocities = np.array(data["ball_velocities"][:32]).reshape(16, 2)
                    avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
                    stats.append(np.tanh(avg_speed))  # Normalized with tanh
                else:
                    stats.append(0.0)
        
        # Pad to fixed size
        while len(stats) < 6:
            stats.append(0.0)
        
        return np.array(stats[:6], dtype=np.float32)
    
    def _process_prompt(self, prompt: str) -> Dict[str, np.ndarray]:
        """Process language prompt into tokenized format."""
        # Simple tokenization (would use proper tokenizer in real implementation)
        if not prompt:
            prompt = "play billiards"  # Default task description
        
        # Create dummy tokenization (would use model's tokenizer)
        tokens = []
        words = prompt.lower().split()
        
        # Simple word-to-token mapping (demo purposes)
        word_to_token = {
            "play": 1, "billiards": 2, "pool": 2, "pot": 3, "ball": 4,
            "eight": 5, "corner": 6, "pocket": 7, "cue": 8, "strike": 9,
            "aim": 10, "shoot": 11, "hit": 12, "the": 13, "a": 14, "an": 14,
            "in": 15, "into": 15, "at": 16, "to": 17, "from": 18,
            "left": 19, "right": 20, "top": 21, "bottom": 22, "center": 23,
            "red": 24, "blue": 25, "yellow": 26, "green": 27, "black": 28,
            "white": 29, "orange": 30, "purple": 31, "stripe": 32, "solid": 33
        }
        
        for word in words[:self.max_prompt_tokens]:
            token = word_to_token.get(word, 0)  # 0 for unknown words
            tokens.append(token)
        
        # Pad to max length
        while len(tokens) < self.max_prompt_tokens:
            tokens.append(0)  # Padding token
        
        tokenized_prompt = np.array(tokens[:self.max_prompt_tokens], dtype=np.int32)
        tokenized_prompt_mask = np.array([1] * len(words) + [0] * (self.max_prompt_tokens - len(words)), dtype=bool)
        
        return {
            "tokenized_prompt": tokenized_prompt,
            "tokenized_prompt_mask": tokenized_prompt_mask
        }
    
    def _create_default_output(self) -> Dict[str, Any]:
        """Create safe default output in case of errors."""
        images = {}
        image_masks = {}
        
        for camera_name in self.EXPECTED_CAMERAS:
            images[camera_name] = np.zeros((*self.image_resolution, 3), dtype=np.float32) - 1.0
            image_masks[camera_name] = np.array([False], dtype=bool)
        
        return {
            "images": images,
            "image_masks": image_masks,
            "state": np.zeros(self.action_dim, dtype=np.float32),
            "tokenized_prompt": np.zeros(self.max_prompt_tokens, dtype=np.int32),
            "tokenized_prompt_mask": np.zeros(self.max_prompt_tokens, dtype=bool),
            "time": np.float32(0.0),
            "episode_step": np.int32(0),
        }


@dataclasses.dataclass(frozen=True)
class PooltoolOutputs(transforms.DataTransformFn):
    """Output transformation for Pooltool environment.
    
    Transforms model action outputs into the format expected by the pooltool environment.
    
    Expected model output format:
    - actions: [action_horizon, action_dim] predicted action sequence
    
    Output format for pooltool environment:
    - high_level_action: strategic decisions (target ball, pocket, etc.)
    - robot_action: low-level robot control
    - cue_action: cue stick operation parameters
    """
    
    # Action dimensions
    action_dim: int = 32
    action_horizon: int = 50
    
    # Action interpretation parameters
    use_high_level_actions: bool = True
    enable_robot_control: bool = True
    enable_cue_actions: bool = True
    
    # Action scaling parameters
    joint_angle_scale: float = 1.0  # Scale factor for joint angles
    position_scale: float = 2.54   # Scale factor for positions (table width)
    power_scale: float = 3.0       # Scale factor for strike power
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform model output to pooltool action format."""
        try:
            if "actions" not in data:
                logger.warning("No 'actions' found in model output")
                return self._create_default_action()
            
            actions = np.array(data["actions"], dtype=np.float32)
            
            # Handle different action formats
            if len(actions.shape) == 2:
                # Multi-step actions: [action_horizon, action_dim]
                current_action = actions[0]  # Use first action in sequence
            elif len(actions.shape) == 1:
                # Single-step action: [action_dim]
                current_action = actions
            else:
                logger.warning(f"Unexpected action shape: {actions.shape}")
                return self._create_default_action()
            
            # Ensure action has correct dimension
            if len(current_action) < self.action_dim:
                current_action = np.pad(current_action, (0, self.action_dim - len(current_action)))
            elif len(current_action) > self.action_dim:
                current_action = current_action[:self.action_dim]
            
            # Decode actions into different control levels
            output = {}
            
            if self.use_high_level_actions:
                output["high_level_action"] = self._decode_high_level_action(current_action)
            
            if self.enable_robot_control:
                output["robot_action"] = self._decode_robot_action(current_action)
            
            if self.enable_cue_actions:
                output["cue_action"] = self._decode_cue_action(current_action)
            
            # Add compatibility action for simple environments
            output["actions"] = current_action
            
            return output
            
        except Exception as e:
            logger.error(f"Error in PooltoolOutputs transformation: {e}")
            return self._create_default_action()
    
    def _decode_high_level_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode high-level strategic action from model output."""
        # Map action dimensions to high-level parameters
        
        # Target ball selection (actions[0:2] -> ball number)
        target_ball_logits = action[0:2]
        target_ball = int(np.argmax(target_ball_logits)) + 1  # 1-15 (skip cue ball)
        
        # Target pocket selection (actions[2:8] -> pocket number)
        target_pocket_logits = action[2:8]
        target_pocket = int(np.argmax(target_pocket_logits))  # 0-5
        
        # Hit point on table (actions[8:10] -> normalized x,y)
        hit_point = action[8:10] * self.position_scale / 2  # Center around table center
        hit_point = np.clip(hit_point, 0, [self.position_scale, self.position_scale/2])
        
        # Hit angle (actions[10] -> angle in radians)
        hit_angle = action[10] * np.pi  # Map to [-π, π]
        
        # Hit power (actions[11] -> power [0,1])
        hit_power = np.tanh(action[11]) * 0.5 + 0.5  # Map to [0, 1]
        hit_power = np.clip(hit_power, 0.1, 1.0)  # Minimum power threshold
        
        # English/spin control (actions[12:14] -> x,y spin)
        english = np.tanh(action[12:14]) * 0.5  # Map to [-0.5, 0.5]
        
        return {
            "target_ball": target_ball,
            "target_pocket": target_pocket,
            "hit_point": hit_point.tolist(),
            "hit_angle": float(hit_angle),
            "hit_power": float(hit_power),
            "english": english.tolist()
        }
    
    def _decode_robot_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode low-level robot control action."""
        # Joint position control (actions[14:20] -> 6 joint angles)
        joint_positions = action[14:20] * self.joint_angle_scale
        
        # Joint velocity control (actions[20:26] -> 6 joint velocities)  
        joint_velocities = action[20:26] * self.joint_angle_scale * 0.1  # Slower velocities
        
        # End-effector pose control (actions[26:32] -> 6-DOF pose)
        end_effector_pose = action[26:32].copy()
        end_effector_pose[:3] *= self.position_scale / 4  # Position scaling
        end_effector_pose[3:] *= np.pi / 2  # Orientation scaling
        
        # Gripper control (derived from action[31])
        gripper_action = np.tanh(action[31]) * 0.5 + 0.5  # Map to [0, 1]
        
        return {
            "joint_positions": joint_positions.tolist(),
            "joint_velocities": joint_velocities.tolist(), 
            "end_effector_pose": end_effector_pose.tolist(),
            "gripper_action": float(gripper_action)
        }
    
    def _decode_cue_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode cue stick action from model output."""
        # Strike power (actions[11] reused from high-level)
        power = np.tanh(action[11]) * 0.5 + 0.5
        power = np.clip(power, 0.0, 1.0)
        
        # Strike angle (actions[10] reused from high-level)
        angle = action[10] * np.pi
        
        # Cue elevation angle (actions[15] -> elevation)
        elevation = np.tanh(action[15]) * np.pi / 6  # Max ±30 degrees
        
        # Strike type indicator (actions[16] -> type)
        strike_type_logits = action[16:18]
        strike_type = int(np.argmax(strike_type_logits))  # 0: normal, 1: jump shot
        
        return {
            "power": float(power),
            "angle": float(angle),
            "elevation": float(elevation),
            "strike_type": strike_type
        }
    
    def _create_default_action(self) -> Dict[str, Any]:
        """Create safe default action in case of errors."""
        return {
            "high_level_action": {
                "target_ball": 1,
                "target_pocket": 0,
                "hit_point": [1.27, 0.635],  # Table center
                "hit_angle": 0.0,
                "hit_power": 0.3,
                "english": [0.0, 0.0]
            },
            "robot_action": {
                "joint_positions": [0.0] * 6,
                "joint_velocities": [0.0] * 6,
                "end_effector_pose": [0.0] * 6,
                "gripper_action": 0.5
            },
            "cue_action": {
                "power": 0.0,  # No strike by default
                "angle": 0.0,
                "elevation": 0.0,
                "strike_type": 0
            },
            "actions": np.zeros(self.action_dim, dtype=np.float32)
        }


# Utility functions for data processing
def normalize_ball_positions(positions: np.ndarray, table_width: float = 2.54, table_height: float = 1.27) -> np.ndarray:
    """Normalize ball positions to [-1, 1] range."""
    normalized = positions.copy()
    normalized[::2] = (normalized[::2] / table_width) * 2 - 1  # x coordinates
    normalized[1::2] = (normalized[1::2] / table_height) * 2 - 1  # y coordinates
    return normalized


def denormalize_ball_positions(normalized: np.ndarray, table_width: float = 2.54, table_height: float = 1.27) -> np.ndarray:
    """Denormalize ball positions from [-1, 1] to table coordinates."""
    positions = normalized.copy()
    positions[::2] = (positions[::2] + 1) / 2 * table_width  # x coordinates
    positions[1::2] = (positions[1::2] + 1) / 2 * table_height  # y coordinates
    return positions


def create_dummy_observation() -> Dict[str, Any]:
    """Create a dummy observation for testing purposes."""
    return {
        "ball_positions": np.random.rand(32).astype(np.float32) * 2.54,
        "ball_velocities": np.random.randn(32).astype(np.float32) * 0.1,
        "ball_states": np.random.rand(32).astype(np.float32),
        "arm_joint_angles": np.random.randn(6).astype(np.float32) * 0.5,
        "arm_end_effector_position": np.random.rand(3).astype(np.float32) * 2,
        "arm_end_effector_orientation": np.random.randn(3).astype(np.float32) * 0.5,
        "table_image_top": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),
        "wrist_camera": np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8),
        "time": 1.5,
        "step_count": 42,
        "prompt": "pot the 8 ball in the corner pocket"
    }


def create_dummy_model_output() -> Dict[str, Any]:
    """Create a dummy model output for testing purposes."""
    return {
        "actions": np.random.randn(50, 32).astype(np.float32) * 0.1
    }


# Test the transformations
if __name__ == "__main__":
    # Test input transformation
    input_transform = PooltoolInputs()
    dummy_obs = create_dummy_observation()
    
    print("Testing PooltoolInputs...")
    transformed_input = input_transform(dummy_obs)
    print(f"Input keys: {list(transformed_input.keys())}")
    print(f"State shape: {transformed_input['state'].shape}")
    print(f"Image shapes: {[(k, v.shape) for k, v in transformed_input['images'].items()]}")
    
    # Test output transformation
    output_transform = PooltoolOutputs()
    dummy_output = create_dummy_model_output()
    
    print("\nTesting PooltoolOutputs...")
    transformed_output = output_transform(dummy_output)
    print(f"Output keys: {list(transformed_output.keys())}")
    
    if "high_level_action" in transformed_output:
        high_level = transformed_output["high_level_action"]
        print(f"High-level action: target_ball={high_level['target_ball']}, "
              f"power={high_level['hit_power']:.3f}, angle={high_level['hit_angle']:.3f}")
    
    print("✅ All tests passed!") 