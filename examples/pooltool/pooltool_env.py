"""
Pooltool Environment for OpenPI
A professional billiards simulation environment with robotic arm integration.

This environment provides:
- Realistic pool table physics using pooltool
- 6-DOF robotic arm control for cue stick manipulation
- Multi-camera observation system
- Hierarchical action space (high-level strategy + low-level control)
- Safety constraints and collision detection
"""

import numpy as np
import pygame
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import time

# Try to import pooltool, fall back to simulation if not available
try:
    import pooltool as pt
    POOLTOOL_AVAILABLE = True
except ImportError:
    POOLTOOL_AVAILABLE = False
    logging.warning("Pooltool not available, using simplified physics simulation")

logger = logging.getLogger(__name__)


@dataclass
class TableConfig:
    """Configuration for the pool table."""
    width: float = 2.54  # meters (9-foot table)
    height: float = 1.27  # meters
    table_height: float = 0.79  # meters from floor
    pocket_radius: float = 0.06  # meters
    ball_radius: float = 0.02625  # meters (52.5mm diameter)
    ball_mass: float = 0.163  # kg


@dataclass
class RobotArmConfig:
    """Configuration for the 6-DOF robotic arm."""
    base_position: Tuple[float, float, float] = (0.0, 0.635, 0.79)  # x, y, z
    max_reach: float = 1.4  # meters
    joint_limits: List[Tuple[float, float]] = None  # [(min, max), ...]
    max_velocity: float = 2.0  # m/s
    end_effector_precision: float = 0.002  # ±2mm
    
    def __post_init__(self):
        if self.joint_limits is None:
            # Default joint limits in radians: [min, max] for each of 6 joints
            self.joint_limits = [
                (-np.pi, np.pi),       # Base rotation
                (-np.pi/2, np.pi/2),   # Shoulder
                (-2*np.pi/3, 2*np.pi/3),  # Elbow
                (-np.pi/2, np.pi/2),   # Wrist pitch
                (-np.pi, np.pi),       # Wrist roll
                (-np.pi, np.pi)        # Tool rotation
            ]


@dataclass
class CameraConfig:
    """Configuration for the multi-camera observation system."""
    # Top camera configuration (table overview)
    top_camera_height: float = 2.5  # meters above table
    top_camera_resolution: Tuple[int, int] = (1920, 1080)
    top_camera_fov: float = 60.0  # Field of view in degrees
    top_camera_position: Tuple[float, float, float] = (1.27, 0.635, 2.5)  # x, y, z
    
    # Side camera configuration (profile view)
    side_camera_height: float = 1.5  # meters
    side_camera_resolution: Tuple[int, int] = (1280, 720)
    side_camera_fov: float = 45.0
    side_camera_position: Tuple[float, float, float] = (-0.5, 0.635, 1.5)  # x, y, z
    side_camera_angle: float = 0.0  # Rotation angle in degrees
    
    # Wrist camera configuration (end-effector mounted)
    wrist_camera_resolution: Tuple[int, int] = (640, 480)
    wrist_camera_fov: float = 75.0  # Wide angle for better coverage
    wrist_camera_offset: Tuple[float, float, float] = (0.05, 0.0, 0.02)  # Offset from end-effector
    
    # Global camera settings
    enable_depth: bool = True  # Enable depth image generation
    enable_noise: bool = True  # Add realistic camera noise
    noise_std: float = 0.01  # Standard deviation of Gaussian noise
    lighting_brightness: float = 1.0  # Global lighting multiplier
    shadow_intensity: float = 0.3  # Shadow darkness (0-1)
    
    # Performance settings
    anti_aliasing: bool = True  # Enable anti-aliasing
    render_quality: str = "high"  # "low", "medium", "high"
    
    def get_camera_intrinsics(self, camera_type: str) -> Dict[str, float]:
        """Get camera intrinsic parameters for a specific camera."""
        if camera_type == "top":
            width, height = self.top_camera_resolution
            fov = self.top_camera_fov
        elif camera_type == "side":
            width, height = self.side_camera_resolution
            fov = self.side_camera_fov
        elif camera_type == "wrist":
            width, height = self.wrist_camera_resolution
            fov = self.wrist_camera_fov
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        
        # Calculate focal length from FOV
        focal_length = (width / 2) / np.tan(np.radians(fov / 2))
        
        return {
            "fx": focal_length,
            "fy": focal_length,
            "cx": width / 2,
            "cy": height / 2,
            "width": width,
            "height": height,
            "fov": fov
        }


class Ball:
    """Represents a billiard ball with physics properties."""
    
    def __init__(self, 
                 position: Tuple[float, float], 
                 number: int = 0,
                 radius: float = 0.02625,
                 mass: float = 0.163):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.number = number
        self.radius = radius
        self.mass = mass
        self.is_pocketed = False
        self.color = self._get_ball_color(number)
        
    def _get_ball_color(self, number: int) -> Tuple[int, int, int]:
        """Get RGB color for ball number."""
        colors = [
            (255, 255, 255),  # 0: Cue ball
            (255, 255, 0),    # 1: Yellow
            (0, 0, 255),      # 2: Blue  
            (255, 0, 0),      # 3: Red
            (128, 0, 128),    # 4: Purple
            (255, 165, 0),    # 5: Orange
            (0, 128, 0),      # 6: Green
            (128, 0, 0),      # 7: Maroon
            (0, 0, 0),        # 8: Black (8-ball)
            (255, 255, 0),    # 9: Yellow stripe
            (0, 0, 255),      # 10: Blue stripe
            (255, 0, 0),      # 11: Red stripe
            (128, 0, 128),    # 12: Purple stripe
            (255, 165, 0),    # 13: Orange stripe
            (0, 128, 0),      # 14: Green stripe
            (128, 0, 0),      # 15: Maroon stripe
        ]
        return colors[min(number, len(colors) - 1)]
    
    def update(self, dt: float, friction: float = 0.02):
        """Update ball physics with friction."""
        if self.is_pocketed:
            return
            
        # Apply rolling friction
        if np.linalg.norm(self.velocity) > 0.01:
            friction_force = -friction * self.velocity / np.linalg.norm(self.velocity)
            self.velocity += friction_force * dt
            
            # Stop if velocity is very small
            if np.linalg.norm(self.velocity) < 0.01:
                self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        # Update position
        self.position += self.velocity * dt


class RobotArm:
    """6-DOF robotic arm for cue stick manipulation with advanced kinematics."""
    
    def __init__(self, config: RobotArmConfig):
        self.config = config
        self.joint_angles = np.zeros(6, dtype=np.float32)
        self.joint_velocities = np.zeros(6, dtype=np.float32)
        self.joint_accelerations = np.zeros(6, dtype=np.float32)
        self.end_effector_position = np.array(config.base_position, dtype=np.float32)
        self.end_effector_orientation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # Roll, pitch, yaw
        
        # Cue stick state
        self.cue_attached = False
        self.cue_angle = 0.0  # Angle relative to table surface
        self.cue_power = 0.0  # Strike power [0, 1]
        self.cue_length = 1.016  # Standard cue length (40 inches)
        
        # Advanced DH parameters for precise 6-DOF arm kinematics
        # [a, alpha, d, theta_offset] for each joint
        self.dh_params = np.array([
            [0.0, 0.0, 0.12, 0.0],           # Base to shoulder (joint 1)
            [0.0, np.pi/2, 0.0, -np.pi/2],  # Shoulder joint (joint 2)
            [0.45, 0.0, 0.0, 0.0],          # Upper arm link (joint 3)
            [0.35, 0.0, 0.12, 0.0],         # Forearm link (joint 4)
            [0.0, np.pi/2, 0.08, 0.0],      # Wrist pitch (joint 5)
            [0.0, -np.pi/2, 0.06, 0.0]      # Wrist roll/tool (joint 6)
        ], dtype=np.float32)
        
        # Link parameters for collision detection
        self.link_radii = np.array([0.08, 0.06, 0.05, 0.04, 0.03, 0.02])  # Cylinder radii for each link
        self.link_lengths = np.array([0.12, 0.15, 0.45, 0.35, 0.08, 0.06])  # Length of each link
        
        # Joint transformation matrices (4x4 homogeneous transforms)
        self.joint_transforms = np.zeros((6, 4, 4), dtype=np.float32)
        self.link_positions = np.zeros((7, 3), dtype=np.float32)  # Base + 6 joint positions
        
        # Initialize base position
        self.link_positions[0] = config.base_position
        
        # Motion planning
        self.trajectory_buffer = []
        self.current_trajectory_step = 0
        self.is_moving = False
        
        # Update forward kinematics
        self.forward_kinematics()
    
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute 4x4 transformation matrix from DH parameters."""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ], dtype=np.float32)
    
    def forward_kinematics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute precise end-effector pose using DH parameters."""
        # Start with identity matrix at base
        T = np.eye(4, dtype=np.float32)
        
        # Apply base position offset
        T[:3, 3] = self.config.base_position
        
        # Store base position
        self.link_positions[0] = self.config.base_position
        
        # Compute transformation for each joint
        for i in range(6):
            # Get DH parameters for this joint
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = self.joint_angles[i] + theta_offset
            
            # Compute joint transformation
            T_joint = self.dh_transform(a, alpha, d, theta)
            self.joint_transforms[i] = T_joint
            
            # Update cumulative transformation
            T = T @ T_joint
            
            # Store joint position
            self.link_positions[i + 1] = T[:3, 3]
        
        # Extract end-effector position and orientation
        self.end_effector_position = T[:3, 3]
        
        # Extract Euler angles from rotation matrix (ZYX convention)
        R = T[:3, :3]
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])  # Roll
            y = np.arctan2(-R[2, 0], sy)      # Pitch
            z = np.arctan2(R[1, 0], R[0, 0])  # Yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        self.end_effector_orientation = np.array([x, y, z], dtype=np.float32)
        
        return self.end_effector_position, self.end_effector_orientation
    
    def inverse_kinematics(self, target_pos: np.ndarray, target_orient: np.ndarray,
                          max_iterations: int = 100, tolerance: float = 1e-4) -> Optional[np.ndarray]:
        """Compute joint angles using numerical inverse kinematics (Jacobian method)."""
        base_pos = np.array(self.config.base_position)
        
        # Check if target is reachable (rough estimate)
        distance = np.linalg.norm(target_pos - base_pos)
        max_reach = sum(self.dh_params[:, 0]) + sum(self.dh_params[:, 2])  # Sum of link lengths
        if distance > max_reach * 0.9:  # 90% of theoretical max reach
            return None
        
        # Initialize with current joint angles
        q = self.joint_angles.copy()
        
        # Target pose vector (position + orientation)
        target_pose = np.concatenate([target_pos, target_orient])
        
        for iteration in range(max_iterations):
            # Compute current pose
            current_pos, current_orient = self._forward_kinematics_from_joints(q)
            current_pose = np.concatenate([current_pos, current_orient])
            
            # Compute error
            error = target_pose - current_pose
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                # Check joint limits
                valid = True
                for i, (min_angle, max_angle) in enumerate(self.config.joint_limits):
                    if q[i] < min_angle or q[i] > max_angle:
                        valid = False
                        break
                
                if valid:
                    return q
                else:
                    # Try to fix joint limit violations
                    for i, (min_angle, max_angle) in enumerate(self.config.joint_limits):
                        q[i] = np.clip(q[i], min_angle, max_angle)
            
            # Compute Jacobian
            J = self._compute_jacobian(q)
            
            # Damped least squares solution
            lambda_damping = 0.01
            J_damped = J.T @ np.linalg.inv(J @ J.T + lambda_damping**2 * np.eye(6))
            
            # Update joint angles
            dq = J_damped @ error
            q += 0.1 * dq  # Small step size for stability
        
        return None  # Failed to converge
    
    def _forward_kinematics_from_joints(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics for given joint angles."""
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = self.config.base_position
        
        for i in range(6):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T_joint = self.dh_transform(a, alpha, d, theta)
            T = T @ T_joint
        
        position = T[:3, 3]
        
        # Extract orientation
        R = T[:3, :3]
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        orientation = np.array([x, y, z], dtype=np.float32)
        return position, orientation
    
    def _compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """Compute 6x6 Jacobian matrix for current joint configuration."""
        J = np.zeros((6, 6), dtype=np.float32)
        epsilon = 1e-6
        
        # Get current pose
        pos_0, orient_0 = self._forward_kinematics_from_joints(joint_angles)
        pose_0 = np.concatenate([pos_0, orient_0])
        
        # Compute numerical Jacobian
        for i in range(6):
            q_plus = joint_angles.copy()
            q_plus[i] += epsilon
            
            pos_plus, orient_plus = self._forward_kinematics_from_joints(q_plus)
            pose_plus = np.concatenate([pos_plus, orient_plus])
            
            J[:, i] = (pose_plus - pose_0) / epsilon
        
        return J
    
    def plan_trajectory(self, target_joints: np.ndarray, duration: float = 2.0, 
                       trajectory_type: str = "polynomial") -> bool:
        """Plan smooth trajectory to target joint configuration."""
        if not self._validate_joint_limits(target_joints):
            return False
        
        current_joints = self.joint_angles.copy()
        steps = int(duration / (1/60.0))  # 60 Hz trajectory
        
        if trajectory_type == "polynomial":
            # 5th order polynomial trajectory (smooth acceleration)
            self.trajectory_buffer = []
            
            for step in range(steps + 1):
                t = step / steps
                
                # 5th order polynomial blending
                s = 6*t**5 - 15*t**4 + 10*t**3
                
                joint_step = current_joints + s * (target_joints - current_joints)
                velocity_step = np.zeros(6)  # Simplified velocity
                
                self.trajectory_buffer.append({
                    'joints': joint_step,
                    'velocities': velocity_step,
                    'time': step * (1/60.0)
                })
        
        elif trajectory_type == "linear":
            # Simple linear interpolation
            self.trajectory_buffer = []
            for step in range(steps + 1):
                t = step / steps
                joint_step = current_joints + t * (target_joints - current_joints)
                velocity_step = (target_joints - current_joints) / duration if step < steps else np.zeros(6)
                
                self.trajectory_buffer.append({
                    'joints': joint_step,
                    'velocities': velocity_step,
                    'time': step * (1/60.0)
                })
        
        self.current_trajectory_step = 0
        self.is_moving = True
        return True
    
    def update_trajectory(self) -> bool:
        """Update robot state along planned trajectory."""
        if not self.is_moving or not self.trajectory_buffer:
            return False
        
        if self.current_trajectory_step >= len(self.trajectory_buffer):
            self.is_moving = False
            return False
        
        # Get current trajectory point
        trajectory_point = self.trajectory_buffer[self.current_trajectory_step]
        
        # Update robot state
        self.joint_angles = trajectory_point['joints']
        self.joint_velocities = trajectory_point['velocities']
        
        # Update forward kinematics
        self.forward_kinematics()
        
        self.current_trajectory_step += 1
        
        return self.is_moving
    
    def check_self_collision(self) -> bool:
        """Check for self-collision between arm links."""
        # Check collision between non-adjacent links
        for i in range(len(self.link_positions) - 1):
            for j in range(i + 2, len(self.link_positions) - 1):
                if self._check_link_collision(i, j):
                    return True
        return False
    
    def _check_link_collision(self, link1_idx: int, link2_idx: int) -> bool:
        """Check collision between two cylindrical links."""
        p1_start = self.link_positions[link1_idx]
        p1_end = self.link_positions[link1_idx + 1]
        r1 = self.link_radii[link1_idx]
        
        p2_start = self.link_positions[link2_idx]
        p2_end = self.link_positions[link2_idx + 1]
        r2 = self.link_radii[link2_idx]
        
        # Compute distance between line segments
        min_distance = self._line_segment_distance(p1_start, p1_end, p2_start, p2_end)
        
        # Check if collision occurs
        return min_distance < (r1 + r2 + 0.01)  # 1cm safety margin
    
    def _line_segment_distance(self, p1: np.ndarray, q1: np.ndarray, 
                             p2: np.ndarray, q2: np.ndarray) -> float:
        """Compute minimum distance between two line segments."""
        # Vector between line segments
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        
        a = np.dot(d1, d1)
        b = np.dot(d1, d2)
        c = np.dot(d2, d2)
        d = np.dot(d1, r)
        e = np.dot(d2, r)
        f = np.dot(r, r)
        
        denom = a * c - b * b
        
        if denom != 0:
            s = np.clip((b * e - c * d) / denom, 0, 1)
            t = np.clip((a * e - b * d) / denom, 0, 1)
        else:
            s = 0
            t = np.clip(-d / a, 0, 1) if a != 0 else 0
        
        # Closest points
        c1 = p1 + s * d1
        c2 = p2 + t * d2
        
        return np.linalg.norm(c1 - c2)
    
    def check_environment_collision(self, balls: List['Ball'], table_config) -> bool:
        """Enhanced collision detection with environment objects."""
        # Check collision with balls
        for ball in balls:
            if not ball.is_pocketed:
                ball_3d_pos = np.array([
                    ball.position[0], 
                    ball.position[1], 
                    table_config.table_height + ball.radius
                ])
                
                # Check collision with each arm link
                for i in range(len(self.link_positions) - 1):
                    if self._check_sphere_cylinder_collision(
                        ball_3d_pos, ball.radius,
                        self.link_positions[i], self.link_positions[i + 1], self.link_radii[i]
                    ):
                        return True
        
        # Check collision with table surface
        table_surface_z = table_config.table_height
        for i in range(len(self.link_positions)):
            if self.link_positions[i][2] < table_surface_z + 0.05:  # 5cm clearance
                return True
        
        # Check workspace boundaries
        for i in range(len(self.link_positions)):
            pos = self.link_positions[i]
            if (pos[0] < -0.5 or pos[0] > table_config.width + 0.5 or
                pos[1] < -0.5 or pos[1] > table_config.height + 0.5):
                return True
        
        return False
    
    def _check_sphere_cylinder_collision(self, sphere_center: np.ndarray, sphere_radius: float,
                                       cylinder_start: np.ndarray, cylinder_end: np.ndarray,
                                       cylinder_radius: float) -> bool:
        """Check collision between sphere and cylinder."""
        # Find closest point on cylinder axis to sphere center
        cylinder_axis = cylinder_end - cylinder_start
        axis_length = np.linalg.norm(cylinder_axis)
        
        if axis_length < 1e-6:
            # Degenerate cylinder (point)
            distance = np.linalg.norm(sphere_center - cylinder_start)
            return distance < (sphere_radius + cylinder_radius)
        
        cylinder_unit = cylinder_axis / axis_length
        to_sphere = sphere_center - cylinder_start
        projection = np.dot(to_sphere, cylinder_unit)
        
        # Clamp projection to cylinder length
        projection = np.clip(projection, 0, axis_length)
        
        # Closest point on cylinder axis
        closest_point = cylinder_start + projection * cylinder_unit
        
        # Distance from sphere center to closest point
        distance = np.linalg.norm(sphere_center - closest_point)
        
        return distance < (sphere_radius + cylinder_radius + 0.02)  # 2cm safety margin
    
    def _validate_joint_limits(self, joint_angles: np.ndarray) -> bool:
        """Check if joint angles are within limits."""
        for i, (min_angle, max_angle) in enumerate(self.config.joint_limits):
            if joint_angles[i] < min_angle or joint_angles[i] > max_angle:
                return False
        return True
    
    def can_reach_position(self, position: np.ndarray) -> bool:
        """Enhanced reachability check using inverse kinematics."""
        # Quick geometric check first
        base_pos = np.array(self.config.base_position)
        distance = np.linalg.norm(position - base_pos)
        
        # Rough reachability estimate
        max_reach = sum(self.dh_params[:, 0]) + sum(self.dh_params[:, 2])
        if distance > max_reach * 0.9:
            return False
        
        # Try inverse kinematics with default orientation
        default_orientation = np.array([0.0, -np.pi/4, 0.0])  # Slight downward angle
        joint_solution = self.inverse_kinematics(position, default_orientation)
        
        return joint_solution is not None
    
    def get_cue_tip_position(self) -> np.ndarray:
        """Get position of cue stick tip when attached to end-effector."""
        if not self.cue_attached:
            return self.end_effector_position
        
        # Cue direction based on end-effector orientation
        cue_direction = np.array([
            np.cos(self.end_effector_orientation[2]) * np.cos(self.end_effector_orientation[1]),
            np.sin(self.end_effector_orientation[2]) * np.cos(self.end_effector_orientation[1]),
            np.sin(self.end_effector_orientation[1])
        ])
        
        # Cue tip is at the end of the cue stick
        cue_tip = self.end_effector_position + self.cue_length * cue_direction
        return cue_tip
    
    def get_arm_state_vector(self) -> np.ndarray:
        """Get comprehensive arm state for observation."""
        state = np.concatenate([
            self.joint_angles,                    # 6 joint angles
            self.joint_velocities,                # 6 joint velocities  
            self.end_effector_position,           # 3 end-effector position
            self.end_effector_orientation,        # 3 end-effector orientation
            [float(self.cue_attached)],           # 1 cue attachment state
            [self.cue_angle],                     # 1 cue angle
            [self.cue_power],                     # 1 cue power
            [float(self.is_moving)]               # 1 motion state
        ])
        return state.astype(np.float32)
    
    def emergency_stop(self):
        """Emergency stop - clear trajectory and freeze arm."""
        self.trajectory_buffer.clear()
        self.current_trajectory_step = 0
        self.is_moving = False
        self.joint_velocities.fill(0.0)
        self.joint_accelerations.fill(0.0)
        logger.warning("Robot arm emergency stop activated")


class Camera:
    """Advanced camera model with realistic rendering capabilities."""
    
    def __init__(self, camera_type: str, config: CameraConfig):
        self.type = camera_type
        self.config = config
        self.intrinsics = config.get_camera_intrinsics(camera_type)
        
        # Camera pose (position and orientation)
        if camera_type == "top":
            self.position = np.array(config.top_camera_position, dtype=np.float32)
            self.look_at = np.array([1.27, 0.635, 0.79], dtype=np.float32)  # Table center
            self.up_vector = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        elif camera_type == "side":
            self.position = np.array(config.side_camera_position, dtype=np.float32)
            self.look_at = np.array([1.27, 0.635, 0.79], dtype=np.float32)  # Table center
            self.up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        elif camera_type == "wrist":
            # Will be updated dynamically based on robot arm pose
            self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.look_at = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.up_vector = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        # Initialize rendering surface
        self.width = self.intrinsics["width"]
        self.height = self.intrinsics["height"]
        self.surface = pygame.Surface((self.width, self.height))
        
        # Depth buffer for depth rendering
        if config.enable_depth:
            self.depth_buffer = np.full((self.height, self.width), np.inf, dtype=np.float32)
        
        # Camera transformation matrices
        self.view_matrix = np.eye(4, dtype=np.float32)
        self.projection_matrix = np.eye(4, dtype=np.float32)
        self.update_matrices()
    
    def update_pose(self, position: np.ndarray, look_at: np.ndarray, up_vector: np.ndarray = None):
        """Update camera pose."""
        self.position = position.copy()
        self.look_at = look_at.copy()
        if up_vector is not None:
            self.up_vector = up_vector.copy()
        self.update_matrices()
    
    def update_matrices(self):
        """Update view and projection matrices."""
        # View matrix (look-at transformation)
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up_vector)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        self.view_matrix = np.array([
            [right[0], right[1], right[2], -np.dot(right, self.position)],
            [up[0], up[1], up[2], -np.dot(up, self.position)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, self.position)],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Projection matrix (perspective)
        fov_rad = np.radians(self.intrinsics["fov"])
        aspect_ratio = self.width / self.height
        near = 0.01
        far = 10.0
        
        f = 1.0 / np.tan(fov_rad / 2)
        self.projection_matrix = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)
    
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int, float]:
        """Transform 3D world position to 2D screen coordinates."""
        # Homogeneous coordinates
        world_homo = np.array([world_pos[0], world_pos[1], world_pos[2], 1.0])
        
        # Apply view transformation
        view_pos = self.view_matrix @ world_homo
        
        # Apply projection transformation
        clip_pos = self.projection_matrix @ view_pos
        
        # Perspective divide
        if clip_pos[3] != 0:
            ndc_pos = clip_pos[:3] / clip_pos[3]
        else:
            return -1, -1, np.inf
        
        # Convert to screen coordinates
        screen_x = int((ndc_pos[0] + 1) * self.width / 2)
        screen_y = int((1 - ndc_pos[1]) * self.height / 2)
        depth = -view_pos[2]  # Distance from camera
        
        return screen_x, screen_y, depth
    
    def is_point_visible(self, world_pos: np.ndarray) -> bool:
        """Check if a point is visible from the camera."""
        screen_x, screen_y, depth = self.world_to_screen(world_pos)
        
        return (0 <= screen_x < self.width and 
                0 <= screen_y < self.height and 
                depth > 0)
    
    def render_ball(self, ball: 'Ball', table_height: float):
        """Render a single ball with proper 3D projection."""
        if ball.is_pocketed:
            return
        
        # Ball 3D position
        ball_3d = np.array([ball.position[0], ball.position[1], table_height + ball.radius])
        
        # Project to screen
        screen_x, screen_y, depth = self.world_to_screen(ball_3d)
        
        if not self.is_point_visible(ball_3d):
            return
        
        # Calculate screen radius based on perspective
        radius_3d = ball.radius
        edge_point = ball_3d + np.array([radius_3d, 0, 0])
        edge_x, edge_y, _ = self.world_to_screen(edge_point)
        screen_radius = max(3, abs(edge_x - screen_x))
        
        # Draw ball with depth-based lighting
        lighting_factor = max(0.3, 1.0 - depth / 5.0)  # Darker with distance
        lit_color = tuple(int(c * lighting_factor) for c in ball.color)
        
        pygame.draw.circle(self.surface, lit_color, (screen_x, screen_y), screen_radius)
        pygame.draw.circle(self.surface, (0, 0, 0), (screen_x, screen_y), screen_radius, 2)
        
        # Add specular highlight
        highlight_pos = (screen_x - screen_radius//3, screen_y - screen_radius//3)
        highlight_radius = max(1, screen_radius // 4)
        pygame.draw.circle(self.surface, (255, 255, 255), highlight_pos, highlight_radius)
        
        # Draw ball number if large enough
        if screen_radius > 10 and ball.number > 0:
            try:
                font_size = max(12, screen_radius)
                if not pygame.font.get_init():
                    pygame.font.init()
                font = pygame.font.Font(None, font_size)
                text = font.render(str(ball.number), True, (255, 255, 255))
                text_rect = text.get_rect(center=(screen_x, screen_y))
                self.surface.blit(text, text_rect)
            except pygame.error:
                # Skip text rendering if font system is not available
                pass
        
        # Update depth buffer if enabled
        if self.config.enable_depth:
            self._update_depth_buffer(screen_x, screen_y, screen_radius, depth)
    
    def render_robot_arm(self, robot_arm: 'RobotArm'):
        """Render robot arm with all links."""
        # Draw each link as a cylinder
        for i in range(len(robot_arm.link_positions) - 1):
            start_pos = robot_arm.link_positions[i]
            end_pos = robot_arm.link_positions[i + 1]
            
            # Check if link is visible
            if not (self.is_point_visible(start_pos) or self.is_point_visible(end_pos)):
                continue
            
            # Project start and end points
            start_x, start_y, start_depth = self.world_to_screen(start_pos)
            end_x, end_y, end_depth = self.world_to_screen(end_pos)
            
            # Draw link as line with thickness based on distance
            avg_depth = (start_depth + end_depth) / 2
            thickness = max(1, int(10 / (1 + avg_depth)))
            
            if (0 <= start_x < self.width and 0 <= start_y < self.height and
                0 <= end_x < self.width and 0 <= end_y < self.height):
                
                # Link color based on index
                link_colors = [(100, 100, 100), (120, 120, 120), (140, 140, 140), 
                              (160, 160, 160), (180, 180, 180), (200, 200, 200)]
                color = link_colors[i % len(link_colors)]
                
                pygame.draw.line(self.surface, color, (start_x, start_y), (end_x, end_y), thickness)
                
                # Draw joint as circle
                joint_radius = max(2, thickness)
                pygame.draw.circle(self.surface, (80, 80, 80), (start_x, start_y), joint_radius)
        
        # Highlight end-effector
        ee_pos = robot_arm.end_effector_position
        if self.is_point_visible(ee_pos):
            ee_x, ee_y, ee_depth = self.world_to_screen(ee_pos)
            ee_radius = max(3, int(15 / (1 + ee_depth)))
            pygame.draw.circle(self.surface, (255, 100, 100), (ee_x, ee_y), ee_radius)
        
        # Draw cue stick if attached
        if robot_arm.cue_attached:
            cue_tip = robot_arm.get_cue_tip_position()
            if self.is_point_visible(ee_pos) and self.is_point_visible(cue_tip):
                tip_x, tip_y, _ = self.world_to_screen(cue_tip)
                ee_x, ee_y, _ = self.world_to_screen(ee_pos)
                pygame.draw.line(self.surface, (139, 69, 19), (ee_x, ee_y), (tip_x, tip_y), 3)
    
    def render_table(self, table_config: TableConfig):
        """Render pool table with proper perspective."""
        # Table corners in 3D
        corners_3d = [
            np.array([0, 0, table_config.table_height]),
            np.array([table_config.width, 0, table_config.table_height]),
            np.array([table_config.width, table_config.height, table_config.table_height]),
            np.array([0, table_config.height, table_config.table_height])
        ]
        
        # Project corners to screen
        screen_corners = []
        for corner in corners_3d:
            x, y, _ = self.world_to_screen(corner)
            screen_corners.append((x, y))
        
        # Draw table surface
        if all(0 <= x < self.width and 0 <= y < self.height for x, y in screen_corners):
            pygame.draw.polygon(self.surface, (0, 120, 0), screen_corners)
            pygame.draw.polygon(self.surface, (139, 69, 19), screen_corners, 8)
        
        # Draw pockets
        pocket_positions_3d = [
            np.array([0, 0, table_config.table_height]),
            np.array([table_config.width/2, 0, table_config.table_height]),
            np.array([table_config.width, 0, table_config.table_height]),
            np.array([0, table_config.height, table_config.table_height]),
            np.array([table_config.width/2, table_config.height, table_config.table_height]),
            np.array([table_config.width, table_config.height, table_config.table_height])
        ]
        
        for pocket_pos in pocket_positions_3d:
            if self.is_point_visible(pocket_pos):
                px, py, depth = self.world_to_screen(pocket_pos)
                pocket_radius = max(5, int(40 / (1 + depth)))
                pygame.draw.circle(self.surface, (0, 0, 0), (px, py), pocket_radius)
    
    def _update_depth_buffer(self, x: int, y: int, radius: int, depth: float):
        """Update depth buffer for a circular region."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx*dx + dy*dy <= radius*radius:
                    px, py = x + dx, y + dy
                    if 0 <= px < self.width and 0 <= py < self.height:
                        self.depth_buffer[py, px] = min(self.depth_buffer[py, px], depth)
    
    def add_noise(self, image: np.ndarray) -> np.ndarray:
        """Add realistic camera noise to the image."""
        if not self.config.enable_noise:
            return image
        
        noise = np.random.normal(0, self.config.noise_std * 255, image.shape)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def get_depth_image(self) -> Optional[np.ndarray]:
        """Get normalized depth image."""
        if not self.config.enable_depth:
            return None
        
        # Normalize depth values to 0-255 range
        finite_depths = self.depth_buffer[self.depth_buffer != np.inf]
        if len(finite_depths) == 0:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        min_depth, max_depth = finite_depths.min(), finite_depths.max()
        if max_depth > min_depth:
            normalized = (self.depth_buffer - min_depth) / (max_depth - min_depth)
            normalized[self.depth_buffer == np.inf] = 1.0
            return (normalized * 255).astype(np.uint8)
        else:
            return np.zeros((self.height, self.width), dtype=np.uint8)
    
    def render(self, balls: List['Ball'], robot_arm: 'RobotArm', table_config: TableConfig) -> np.ndarray:
        """Render complete scene and return RGB image."""
        # Clear surface
        self.surface.fill((30, 30, 30))  # Dark background
        
        # Reset depth buffer
        if self.config.enable_depth:
            self.depth_buffer.fill(np.inf)
        
        # Render scene components
        self.render_table(table_config)
        self.render_robot_arm(robot_arm)
        
        for ball in balls:
            self.render_ball(ball, table_config.table_height)
        
        # Convert surface to numpy array
        image = pygame.surfarray.array3d(self.surface)
        image = np.transpose(image, (1, 0, 2))
        
        # Apply post-processing
        image = self.add_noise(image)
        
        # Apply lighting
        if self.config.lighting_brightness != 1.0:
            image = (image * self.config.lighting_brightness).astype(np.uint8)
        
        return image


class PooltoolEnvironment(_environment.Environment):
    """Professional pool table environment with robotic arm integration."""
    
    def __init__(self,
                 table_config: Optional[TableConfig] = None,
                 robot_config: Optional[RobotArmConfig] = None,
                 camera_config: Optional[CameraConfig] = None,
                 num_balls: int = 16,
                 render_mode: str = "rgb_array",
                 max_episode_steps: int = 1000,
                 seed: int = 42):
        """
        Initialize the Pooltool environment.
        
        Args:
            table_config: Table configuration parameters
            robot_config: Robot arm configuration parameters  
            camera_config: Camera system configuration
            num_balls: Number of balls (1 cue + 15 object balls)
            render_mode: "human" or "rgb_array"
            max_episode_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        # Configuration
        self.table_config = table_config or TableConfig()
        self.robot_config = robot_config or RobotArmConfig()
        self.camera_config = camera_config or CameraConfig()
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        
        # Random number generator
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        
        # Environment components
        self.robot_arm = RobotArm(self.robot_config)
        self.balls: List[Ball] = []
        self._setup_balls(num_balls)
        
        # Physics simulation
        self.dt = 1/60.0  # 60 FPS
        self.time = 0.0
        self.step_count = 0
        
        # Episode state
        self._episode_complete = False
        self._last_observation = None
        
        # Rendering
        self.screen = None
        self.clock = None
        self.screen_width = 1200
        self.screen_height = 600
        
        # Initialize advanced camera system
        self.cameras = {
            "top": Camera("top", self.camera_config),
            "side": Camera("side", self.camera_config),
            "wrist": Camera("wrist", self.camera_config)
        }
        
        # Pooltool integration
        if POOLTOOL_AVAILABLE:
            self._init_pooltool()
        else:
            logger.info("Using simplified physics simulation")
            
        logger.info(f"PooltoolEnvironment initialized with {len(self.balls)} balls and {len(self.cameras)} cameras")
    
    def _init_pooltool(self):
        """Initialize pooltool physics engine."""
        try:
            # Create pooltool table and system
            self.pt_table = pt.Table.default()
            self.pt_system = pt.System(
                table=self.pt_table,
                balls=[]
            )
            logger.info("Pooltool physics engine initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize pooltool: {e}")
            global POOLTOOL_AVAILABLE
            POOLTOOL_AVAILABLE = False
    
    def _setup_balls(self, num_balls: int):
        """Setup initial ball positions in standard rack formation."""
        self.balls = []
        
        # Cue ball position
        cue_ball = Ball(
            position=(self.table_config.width * 0.25, self.table_config.height * 0.5),
            number=0
        )
        self.balls.append(cue_ball)
        
        # Object balls in triangle rack
        if num_balls > 1:
            self._setup_rack_formation(num_balls - 1)
        
        logger.info(f"Setup {len(self.balls)} balls on table")
    
    def _setup_rack_formation(self, num_object_balls: int):
        """Setup object balls in triangle rack formation."""
        rack_start_x = self.table_config.width * 0.75
        rack_start_y = self.table_config.height * 0.5
        ball_diameter = 2 * self.table_config.ball_radius
        
        ball_number = 1
        for row in range(5):  # 5 rows in triangle rack
            balls_in_row = row + 1
            for col in range(balls_in_row):
                if ball_number > num_object_balls:
                    break
                
                # Hexagonal close packing
                x = rack_start_x + row * ball_diameter * 0.866
                y = rack_start_y + (col - balls_in_row/2 + 0.5) * ball_diameter
                
                ball = Ball(position=(x, y), number=ball_number)
                self.balls.append(ball)
                ball_number += 1
                
            if ball_number > num_object_balls:
                break
    
    @override
    def reset(self) -> None:
        """Reset environment to initial state."""
        logger.info("Resetting environment")
        
        # Reset balls
        self._setup_balls(len(self.balls))
        
        # Reset robot arm
        self.robot_arm = RobotArm(self.robot_config)
        
        # Reset episode state
        self.time = 0.0
        self.step_count = 0
        self._episode_complete = False
        self._last_observation = None
        
        # Get initial observation
        self._last_observation = self._get_observation()
        
        logger.info("Environment reset complete")
    
    @override
    def is_episode_complete(self) -> bool:
        """Check if episode is complete."""
        # Episode ends if:
        # 1. Maximum steps reached
        # 2. All balls stopped and no valid shots
        # 3. Game objective completed (8-ball pocketed, etc.)
        
        if self.step_count >= self.max_episode_steps:
            logger.info("Episode complete: Maximum steps reached")
            return True
        
        if self._all_balls_stopped():
            # Check if 8-ball is pocketed (basic win condition)
            eight_ball = next((ball for ball in self.balls if ball.number == 8), None)
            if eight_ball and eight_ball.is_pocketed:
                logger.info("Episode complete: 8-ball pocketed")
                return True
        
        return self._episode_complete
    
    @override
    def get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        if self._last_observation is None:
            self._last_observation = self._get_observation()
        return self._last_observation
    
    def _get_observation(self) -> Dict[str, Any]:
        """Generate comprehensive observation from current state."""
        # Ball state information
        ball_positions = []
        ball_velocities = []
        ball_states = []  # [is_pocketed, number]
        
        for ball in self.balls:
            ball_positions.extend([ball.position[0], ball.position[1]])
            ball_velocities.extend([ball.velocity[0], ball.velocity[1]])
            ball_states.extend([float(ball.is_pocketed), float(ball.number)])
        
        # Pad to handle variable number of balls
        max_balls = 16
        while len(ball_positions) < max_balls * 2:
            ball_positions.extend([0.0, 0.0])
            ball_velocities.extend([0.0, 0.0])
            ball_states.extend([1.0, -1.0])  # Pocketed, invalid number
        
        # Robot arm state
        arm_position, arm_orientation = self.robot_arm.forward_kinematics()
        arm_state = np.concatenate([
            self.robot_arm.joint_angles,
            self.robot_arm.joint_velocities,
            arm_position,
            arm_orientation,
            [self.robot_arm.cue_angle, self.robot_arm.cue_power, float(self.robot_arm.cue_attached)]
        ])
        
        # Update wrist camera position based on robot arm pose
        self._update_wrist_camera_pose()
        
        # Generate camera images using enhanced camera system
        table_image_top = self.cameras["top"].render(self.balls, self.robot_arm, self.table_config)
        table_image_side = self.cameras["side"].render(self.balls, self.robot_arm, self.table_config)
        wrist_camera = self.cameras["wrist"].render(self.balls, self.robot_arm, self.table_config)
        
        # Additional state information
        table_state = np.array([
            self.time,
            float(self.step_count),
            float(self._all_balls_stopped()),
            float(self._episode_complete)
        ], dtype=np.float32)
        
        observation = {
            # Ball information
            "ball_positions": np.array(ball_positions, dtype=np.float32),
            "ball_velocities": np.array(ball_velocities, dtype=np.float32),
            "ball_states": np.array(ball_states, dtype=np.float32),
            
            # Robot arm state
            "arm_joint_angles": self.robot_arm.joint_angles.astype(np.float32),
            "arm_joint_velocities": self.robot_arm.joint_velocities.astype(np.float32),
            "arm_end_effector_position": arm_position.astype(np.float32),
            "arm_end_effector_orientation": arm_orientation.astype(np.float32),
            "arm_state": arm_state.astype(np.float32),
            
            # Camera observations
            "table_image_top": table_image_top,
            "table_image_side": table_image_side,
            "wrist_camera": wrist_camera,
            
            # Environment state
            "table_state": table_state,
            "time": np.float32(self.time),
            "step_count": np.int32(self.step_count),
            
            # For compatibility with openpi models
            "base_0_rgb": table_image_top,
            "left_wrist_0_rgb": wrist_camera,
            "right_wrist_0_rgb": wrist_camera,
            "state": arm_state[:32],  # Truncate to match expected state dimension
        }
        
        return observation
    
    @override
    def apply_action(self, action: Dict[str, Any]) -> None:
        """Apply action to environment."""
        logger.debug(f"Applying action at step {self.step_count}")
        
        # Handle different action formats
        if "high_level_action" in action:
            self._apply_high_level_action(action["high_level_action"])
        
        if "robot_action" in action:
            self._apply_robot_action(action["robot_action"])
            
        if "cue_action" in action:
            self._apply_cue_action(action["cue_action"])
        
        # Handle simple action format for compatibility
        if "actions" in action:
            self._apply_joint_action(action["actions"])
        
        # Update physics simulation
        self._update_physics()
        
        # Update step counter and time
        self.step_count += 1
        self.time += self.dt
        
        # Generate new observation
        self._last_observation = self._get_observation()
        
        logger.debug(f"Step {self.step_count} completed, time: {self.time:.3f}s")
    
    def _apply_high_level_action(self, action: Dict[str, Any]):
        """Apply high-level strategic action."""
        target_ball = action.get("target_ball", 1)
        target_pocket = action.get("target_pocket", 0)
        hit_point = action.get("hit_point", [0.0, 0.0])
        hit_angle = action.get("hit_angle", 0.0)
        hit_power = action.get("hit_power", 0.5)
        
        # Convert high-level action to robot motion
        self._plan_shot(target_ball, target_pocket, hit_point, hit_angle, hit_power)
    
    def _apply_robot_action(self, action: Dict[str, Any]):
        """Apply robot arm control action."""
        if "joint_positions" in action:
            target_joints = np.array(action["joint_positions"], dtype=np.float32)
            self._move_to_joint_positions(target_joints)
        
        if "end_effector_pose" in action:
            target_pose = np.array(action["end_effector_pose"], dtype=np.float32)
            self._move_to_end_effector_pose(target_pose)
        
        if "gripper_action" in action:
            gripper_value = float(action["gripper_action"])
            self.robot_arm.cue_attached = gripper_value > 0.5
    
    def _apply_cue_action(self, action: Dict[str, Any]):
        """Apply cue stick action."""
        power = action.get("power", 0.0)
        angle = action.get("angle", 0.0)
        
        if power > 0.1:  # Minimum power threshold
            self._strike_cue_ball(power, angle)
    
    def _apply_joint_action(self, actions: Union[np.ndarray, List[float]]):
        """Apply joint-level action for compatibility."""
        if isinstance(actions, list):
            actions = np.array(actions, dtype=np.float32)
        
        # Take first 6 values as joint angles
        if len(actions) >= 6:
            self.robot_arm.joint_angles = actions[:6]
            self.robot_arm.forward_kinematics()
    
    def _plan_shot(self, target_ball: int, target_pocket: int, 
                   hit_point: List[float], hit_angle: float, hit_power: float):
        """Plan and execute a shot based on high-level parameters."""
        # Find target ball
        target_ball_obj = next((ball for ball in self.balls if ball.number == target_ball), None)
        if not target_ball_obj or target_ball_obj.is_pocketed:
            return
        
        # Calculate optimal striking position
        ball_pos = target_ball_obj.position
        striking_distance = 0.15  # 15cm behind ball
        
        strike_pos = np.array([
            ball_pos[0] - striking_distance * np.cos(hit_angle),
            ball_pos[1] - striking_distance * np.sin(hit_angle),
            self.table_config.table_height + 0.05  # 5cm above table
        ])
        
        # Move robot arm to striking position
        if self.robot_arm.can_reach_position(strike_pos):
            target_orientation = np.array([0.0, hit_angle, 0.0])
            joint_angles = self.robot_arm.inverse_kinematics(strike_pos, target_orientation)
            
            if joint_angles is not None:
                self.robot_arm.joint_angles = joint_angles
                self.robot_arm.cue_power = hit_power
                self.robot_arm.cue_angle = hit_angle
                self.robot_arm.forward_kinematics()
    
    def _move_to_joint_positions(self, target_joints: np.ndarray):
        """Move robot arm to target joint positions with collision checking."""
        # Apply joint limits
        for i, (min_angle, max_angle) in enumerate(self.robot_config.joint_limits):
            if i < len(target_joints):
                target_joints[i] = np.clip(target_joints[i], min_angle, max_angle)
        
        # Store current state
        current_joints = self.robot_arm.joint_angles.copy()
        
        # Test target configuration
        self.robot_arm.joint_angles = target_joints[:6]
        self.robot_arm.forward_kinematics()
        
        # Check for collisions at target position
        if self.robot_arm.check_environment_collision(self.balls, self.table_config):
            logger.warning("Target joint position would cause environment collision")
            # Restore previous configuration
            self.robot_arm.joint_angles = current_joints
            self.robot_arm.forward_kinematics()
            return
        
        if self.robot_arm.check_self_collision():
            logger.warning("Target joint position would cause self-collision")
            # Restore previous configuration
            self.robot_arm.joint_angles = current_joints
            self.robot_arm.forward_kinematics()
            return
        
        # Use trajectory planning for smooth motion
        if not self.robot_arm.plan_trajectory(target_joints[:6], duration=1.0):
            logger.warning("Failed to plan trajectory to target joints")
            self.robot_arm.joint_angles = current_joints
            self.robot_arm.forward_kinematics()
    
    def _move_to_end_effector_pose(self, target_pose: np.ndarray):
        """Move robot arm to target end-effector pose with collision checking."""
        if len(target_pose) >= 6:
            target_pos = target_pose[:3]
            target_orient = target_pose[3:6]
            
            # Check if target position is reachable
            if not self.robot_arm.can_reach_position(target_pos):
                logger.warning("Target end-effector position is not reachable")
                return
            
            # Compute inverse kinematics
            joint_angles = self.robot_arm.inverse_kinematics(target_pos, target_orient)
            if joint_angles is not None:
                # Use the enhanced joint movement with collision checking
                self._move_to_joint_positions(joint_angles)
            else:
                logger.warning("Inverse kinematics failed for target pose")
    
    def _strike_cue_ball(self, power: float, angle: float):
        """Execute cue ball strike."""
        cue_ball = self.balls[0]  # First ball is always cue ball
        
        if not cue_ball.is_pocketed and self._all_balls_stopped():
            # Apply force to cue ball
            velocity_magnitude = power * 3.0  # Scale factor
            cue_ball.velocity = np.array([
                velocity_magnitude * np.cos(angle),
                velocity_magnitude * np.sin(angle)
            ], dtype=np.float32)
            
            logger.info(f"Cue ball struck with power {power:.2f} at angle {angle:.2f} rad")
    
    def _update_physics(self):
        """Update physics simulation for one time step."""
        # Update robot arm trajectory
        self.robot_arm.update_trajectory()
        
        # Update ball physics
        if POOLTOOL_AVAILABLE and hasattr(self, 'pt_system'):
            self._update_pooltool_physics()
        else:
            self._update_simple_physics()
    
    def _update_pooltool_physics(self):
        """Update physics using pooltool engine."""
        try:
            # Sync ball states with pooltool
            # This would require more complex integration
            # For now, fall back to simple physics
            self._update_simple_physics()
        except Exception as e:
            logger.warning(f"Pooltool physics update failed: {e}")
            self._update_simple_physics()
    
    def _update_simple_physics(self):
        """Update physics using simplified simulation."""
        # Update ball physics
        for ball in self.balls:
            ball.update(self.dt)
        
        # Handle ball-ball collisions
        self._handle_ball_collisions()
        
        # Handle ball-table collisions
        self._handle_boundary_collisions()
        
        # Check for pocketed balls
        self._check_pockets()
    
    def _handle_ball_collisions(self):
        """Handle elastic collisions between balls."""
        for i, ball1 in enumerate(self.balls):
            if ball1.is_pocketed:
                continue
                
            for j, ball2 in enumerate(self.balls[i+1:], i+1):
                if ball2.is_pocketed:
                    continue
                
                distance = np.linalg.norm(ball1.position - ball2.position)
                min_distance = ball1.radius + ball2.radius
                
                if distance < min_distance:
                    self._resolve_ball_collision(ball1, ball2)
    
    def _resolve_ball_collision(self, ball1: Ball, ball2: Ball):
        """Resolve elastic collision between two balls."""
        # Collision normal vector
        dx = ball2.position[0] - ball1.position[0]
        dy = ball2.position[1] - ball1.position[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:  # Avoid division by zero
            return
        
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        dvx = ball2.velocity[0] - ball1.velocity[0]
        dvy = ball2.velocity[1] - ball1.velocity[1]
        
        # Relative velocity in collision normal direction
        dvn = dvx * nx + dvy * ny
        
        # Don't resolve if velocities are separating
        if dvn > 0:
            return
        
        # Collision impulse (assuming equal mass)
        impulse = 2 * dvn / (ball1.mass + ball2.mass)
        
        # Update velocities
        impulse_x = impulse * ball2.mass * nx
        impulse_y = impulse * ball2.mass * ny
        
        ball1.velocity[0] += impulse_x
        ball1.velocity[1] += impulse_y
        ball2.velocity[0] -= impulse_x
        ball2.velocity[1] -= impulse_y
        
        # Separate overlapping balls
        overlap = (ball1.radius + ball2.radius) - distance
        if overlap > 0:
            separation = overlap / 2
            ball1.position[0] -= separation * nx
            ball1.position[1] -= separation * ny
            ball2.position[0] += separation * nx
            ball2.position[1] += separation * ny
    
    def _handle_boundary_collisions(self):
        """Handle collisions with table boundaries."""
        for ball in self.balls:
            if ball.is_pocketed:
                continue
            
            # Left/right cushions
            if ball.position[0] - ball.radius < 0:
                ball.position[0] = ball.radius
                ball.velocity[0] = -ball.velocity[0] * 0.8  # Energy loss
            elif ball.position[0] + ball.radius > self.table_config.width:
                ball.position[0] = self.table_config.width - ball.radius
                ball.velocity[0] = -ball.velocity[0] * 0.8
            
            # Top/bottom cushions
            if ball.position[1] - ball.radius < 0:
                ball.position[1] = ball.radius
                ball.velocity[1] = -ball.velocity[1] * 0.8
            elif ball.position[1] + ball.radius > self.table_config.height:
                ball.position[1] = self.table_config.height - ball.radius
                ball.velocity[1] = -ball.velocity[1] * 0.8
    
    def _check_pockets(self):
        """Check if any balls have fallen into pockets."""
        # Pocket positions (6 pockets total)
        pockets = [
            (0, 0),  # Bottom left corner
            (self.table_config.width/2, 0),  # Bottom center
            (self.table_config.width, 0),  # Bottom right corner
            (0, self.table_config.height),  # Top left corner  
            (self.table_config.width/2, self.table_config.height),  # Top center
            (self.table_config.width, self.table_config.height)  # Top right corner
        ]
        
        for ball in self.balls:
            if ball.is_pocketed:
                continue
            
            for pocket_x, pocket_y in pockets:
                distance = np.sqrt((ball.position[0] - pocket_x)**2 + (ball.position[1] - pocket_y)**2)
                if distance < self.table_config.pocket_radius:
                    ball.is_pocketed = True
                    ball.velocity = np.array([0.0, 0.0], dtype=np.float32)
                    logger.info(f"Ball {ball.number} pocketed")
                    break
    
    def _all_balls_stopped(self) -> bool:
        """Check if all balls have stopped moving."""
        for ball in self.balls:
            if not ball.is_pocketed and np.linalg.norm(ball.velocity) > 0.01:
                return False
        return True
    
    def _render_top_view(self) -> np.ndarray:
        """Render top-down view of the table."""
        height, width = self.camera_config.top_camera_resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if self.render_mode == "rgb_array":
            # Create pygame surface for rendering
            surface = pygame.Surface((width, height))
            surface.fill((0, 100, 0))  # Green table
            
            # Scale factors
            scale_x = width / self.table_config.width
            scale_y = height / self.table_config.height
            
            # Draw table boundaries
            pygame.draw.rect(surface, (139, 69, 19), (0, 0, width, height), 10)
            
            # Draw pockets
            pockets = [
                (0, 0), (width//2, 0), (width, 0),
                (0, height), (width//2, height), (width, height)
            ]
            for px, py in pockets:
                pygame.draw.circle(surface, (0, 0, 0), (px, py), 20)
            
            # Draw balls
            for ball in self.balls:
                if not ball.is_pocketed:
                    screen_x = int(ball.position[0] * scale_x)
                    screen_y = int(ball.position[1] * scale_y)
                    screen_radius = int(ball.radius * min(scale_x, scale_y))
                    
                    pygame.draw.circle(surface, ball.color, (screen_x, screen_y), screen_radius)
                    pygame.draw.circle(surface, (0, 0, 0), (screen_x, screen_y), screen_radius, 2)
                    
                    # Draw ball number
                    if ball.number > 0:
                        font = pygame.font.Font(None, 24)
                        text = font.render(str(ball.number), True, (255, 255, 255))
                        text_rect = text.get_rect(center=(screen_x, screen_y))
                        surface.blit(text, text_rect)
            
            # Draw robot arm (simplified)
            arm_pos = self.robot_arm.end_effector_position
            if arm_pos[2] > self.table_config.table_height - 0.1:  # If close to table
                arm_x = int(arm_pos[0] * scale_x)
                arm_y = int(arm_pos[1] * scale_y)
                pygame.draw.circle(surface, (200, 200, 200), (arm_x, arm_y), 15)
                
                # Draw base position
                base_x = int(self.robot_config.base_position[0] * scale_x)
                base_y = int(self.robot_config.base_position[1] * scale_y)
                pygame.draw.circle(surface, (100, 100, 100), (base_x, base_y), 10)
                pygame.draw.line(surface, (150, 150, 150), (base_x, base_y), (arm_x, arm_y), 3)
            
            # Convert surface to numpy array
            image = pygame.surfarray.array3d(surface)
            image = np.transpose(image, (1, 0, 2))
        
        return image
    
    def _render_side_view(self) -> np.ndarray:
        """Render side view of the table and robot arm."""
        height, width = self.camera_config.side_camera_resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simplified side view rendering
        image[:, :] = [50, 50, 50]  # Dark background
        
        # Draw table silhouette
        table_y = int(height * 0.7)  # Table at 70% height
        image[table_y:table_y+20, :] = [139, 69, 19]  # Brown table
        
        # Draw robot arm silhouette
        arm_pos = self.robot_arm.end_effector_position
        arm_x = int(width * arm_pos[0] / self.table_config.width)
        arm_z = int(height * (1 - arm_pos[2] / 2.0))  # Normalized height
        
        if 0 <= arm_x < width and 0 <= arm_z < height:
            image[max(0, arm_z-10):min(height, arm_z+10), 
                  max(0, arm_x-5):min(width, arm_x+5)] = [200, 200, 200]
        
        return image
    
    def _render_wrist_view(self) -> np.ndarray:
        """Render view from robot arm wrist camera."""
        height, width = self.camera_config.wrist_camera_resolution
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simplified wrist camera view
        # Show balls in field of view
        arm_pos = self.robot_arm.end_effector_position
        
        for ball in self.balls:
            if not ball.is_pocketed:
                # Calculate ball position relative to wrist camera
                ball_3d = np.array([ball.position[0], ball.position[1], self.table_config.table_height])
                relative_pos = ball_3d - arm_pos
                
                # Simple perspective projection
                if relative_pos[2] != 0 and np.linalg.norm(relative_pos) < 0.5:  # Within 50cm
                    proj_x = int(width/2 + relative_pos[0] * 200)
                    proj_y = int(height/2 + relative_pos[1] * 200)
                    
                    if 0 <= proj_x < width and 0 <= proj_y < height:
                        radius = max(5, int(20 / (1 + np.linalg.norm(relative_pos))))
                        
                        # Draw ball in image
                        y_min = max(0, proj_y - radius)
                        y_max = min(height, proj_y + radius)
                        x_min = max(0, proj_x - radius)
                        x_max = min(width, proj_x + radius)
                        
                        image[y_min:y_max, x_min:x_max] = ball.color
        
        return image
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Pooltool Environment")
                self.clock = pygame.time.Clock()
            
            # Render top view to screen
            top_view = self._render_top_view()
            resized = pygame.transform.scale(
                pygame.surfarray.make_surface(np.transpose(top_view, (1, 0, 2))),
                (self.screen_width, self.screen_height)
            )
            self.screen.blit(resized, (0, 0))
            
            pygame.display.flip()
            self.clock.tick(60)
            
        elif self.render_mode == "rgb_array":
            return self._render_top_view()
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
        
        logger.info("Environment closed")

    def _update_wrist_camera_pose(self):
        """Update wrist camera position based on current robot arm pose."""
        # Get end-effector position and orientation
        ee_pos = self.robot_arm.end_effector_position
        ee_orient = self.robot_arm.end_effector_orientation
        
        # Apply camera offset from end-effector
        offset = np.array(self.camera_config.wrist_camera_offset)
        
        # Rotate offset based on end-effector orientation
        # Simple rotation around Z-axis for now
        cos_z = np.cos(ee_orient[2])
        sin_z = np.sin(ee_orient[2])
        
        rotated_offset = np.array([
            offset[0] * cos_z - offset[1] * sin_z,
            offset[0] * sin_z + offset[1] * cos_z,
            offset[2]
        ])
        
        camera_pos = ee_pos + rotated_offset
        
        # Look direction: slightly forward and down from end-effector
        look_direction = np.array([
            np.cos(ee_orient[2]) * np.cos(ee_orient[1]),
            np.sin(ee_orient[2]) * np.cos(ee_orient[1]),
            np.sin(ee_orient[1]) - 0.2  # Slight downward tilt
        ])
        
        look_at = camera_pos + look_direction * 0.5  # Look 50cm ahead
        
        # Update camera pose
        self.cameras["wrist"].update_pose(camera_pos, look_at)


def make_example() -> Dict[str, Any]:
    """Create a random input example for testing."""
    return {
        "ball_positions": np.random.rand(32).astype(np.float32),
        "ball_velocities": np.random.rand(32).astype(np.float32),
        "ball_states": np.random.rand(32).astype(np.float32),
        "arm_joint_angles": np.random.rand(6).astype(np.float32),
        "arm_state": np.random.rand(15).astype(np.float32),
        "table_image_top": np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8),
        "base_0_rgb": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "right_wrist_0_rgb": np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
        "state": np.random.rand(32).astype(np.float32),
        "time": np.float32(0.0),
        "prompt": "pot the 8 ball in the corner pocket"
    } 