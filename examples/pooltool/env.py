"""
Pooltool Environment for OpenPI
A billiards simulation environment with robotic arm integration.
"""

import numpy as np
import pygame
import pymunk
from typing import Dict, List, Tuple, Optional, Any
from openpi_client.runtime import environment as _environment
from typing_extensions import override


class Ball:
    """Represents a billiard ball with physics properties."""
    
    def __init__(self, pos: Tuple[float, float], radius: float = 0.028, mass: float = 0.16):
        self.radius = radius  # Standard pool ball radius in meters
        self.mass = mass      # Standard pool ball mass in kg
        self.position = np.array(pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.number = 0       # Ball number (0=cue ball, 1-15=object balls)
        self.color = (255, 255, 255)  # RGB color
        
    def update(self, dt: float, friction: float = 0.02):
        """Update ball position and apply friction."""
        # Apply friction
        if np.linalg.norm(self.velocity) > 0:
            friction_force = -friction * self.velocity / np.linalg.norm(self.velocity)
            self.velocity += friction_force * dt
            
            # Stop if velocity becomes very small
            if np.linalg.norm(self.velocity) < 0.01:
                self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        
        # Update position
        self.position += self.velocity * dt


class Table:
    """Represents a pool table with boundaries and pockets."""
    
    def __init__(self, width: float = 2.54, height: float = 1.27):
        self.width = width    # Standard 9-foot table in meters
        self.height = height
        self.pockets = [
            (0, 0), (width/2, 0), (width, 0),           # Top pockets
            (0, height), (width/2, height), (width, height)  # Bottom pockets
        ]
        self.pocket_radius = 0.06  # Pocket radius in meters


class RobotArm:
    """Simplified robot arm model for pool cue manipulation."""
    
    def __init__(self, base_pos: Tuple[float, float] = (1.27, 0.635)):
        self.base_position = np.array(base_pos, dtype=np.float32)  # Center of table
        self.joint_angles = np.zeros(6, dtype=np.float32)  # 6-DOF arm
        self.end_effector_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 3D position
        self.cue_angle = 0.0  # Angle of the cue stick
        self.cue_power = 0.0  # Power for striking the ball
        
    def forward_kinematics(self) -> np.ndarray:
        """Simple forward kinematics calculation."""
        # Simplified 2D projection for demonstration
        reach = 0.8  # Maximum reach in meters
        x = self.base_position[0] + reach * np.cos(self.joint_angles[0])
        y = self.base_position[1] + reach * np.sin(self.joint_angles[0])
        z = 0.75 + 0.3 * np.sin(self.joint_angles[1])  # Height above table
        
        self.end_effector_pos = np.array([x, y, z], dtype=np.float32)
        return self.end_effector_pos
    
    def can_reach_ball(self, ball_pos: Tuple[float, float]) -> bool:
        """Check if the arm can reach a ball position."""
        distance = np.linalg.norm(np.array(ball_pos) - self.base_position)
        return distance <= 0.8  # Maximum reach


class PooltoolEnvironment(_environment.Environment):
    """Pool table environment with robotic arm integration."""
    
    def __init__(self, 
                 table_size: Tuple[float, float] = (2.54, 1.27),
                 num_balls: int = 16,
                 render_mode: str = "human",
                 seed: int = 0):
        """
        Initialize the pool environment.
        
        Args:
            table_size: (width, height) of the table in meters
            num_balls: Number of balls to simulate (1 cue + 15 object balls)
            render_mode: Rendering mode ("human" or "rgb_array")
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        self._rng = np.random.default_rng(seed)
        
        # Environment setup
        self.table = Table(table_size[0], table_size[1])
        self.robot_arm = RobotArm()
        self.render_mode = render_mode
        
        # Initialize balls
        self.balls: List[Ball] = []
        self._setup_balls(num_balls)
        
        # Physics simulation
        self.dt = 1/60.0  # 60 FPS simulation
        self.time = 0.0
        
        # Rendering
        self.screen = None
        self.clock = None
        self.screen_width = 800
        self.screen_height = 400
        
        # Environment state
        self._last_obs = None
        self._done = False
        self._episode_reward = 0.0
        
    def _setup_balls(self, num_balls: int):
        """Setup initial ball positions in rack formation."""
        self.balls = []
        
        # Cue ball
        cue_ball = Ball((self.table.width * 0.25, self.table.height * 0.5))
        cue_ball.number = 0
        cue_ball.color = (255, 255, 255)  # White
        self.balls.append(cue_ball)
        
        # Object balls in rack formation
        if num_balls > 1:
            rack_start_x = self.table.width * 0.75
            rack_start_y = self.table.height * 0.5
            ball_diameter = 0.056  # 2 * radius
            
            ball_num = 1
            for row in range(5):  # 5 rows in standard rack
                balls_in_row = row + 1
                for col in range(balls_in_row):
                    if ball_num >= num_balls:
                        break
                        
                    x = rack_start_x + row * ball_diameter * 0.866  # hex packing
                    y = rack_start_y + (col - balls_in_row/2 + 0.5) * ball_diameter
                    
                    ball = Ball((x, y))
                    ball.number = ball_num
                    ball.color = self._get_ball_color(ball_num)
                    self.balls.append(ball)
                    
                    ball_num += 1
                if ball_num >= num_balls:
                    break
                    
    def _get_ball_color(self, ball_number: int) -> Tuple[int, int, int]:
        """Get color for numbered ball."""
        colors = [
            (255, 255, 255),  # 0: Cue ball (white)
            (255, 255, 0),    # 1: Yellow
            (0, 0, 255),      # 2: Blue
            (255, 0, 0),      # 3: Red
            (128, 0, 128),    # 4: Purple
            (255, 165, 0),    # 5: Orange
            (0, 128, 0),      # 6: Green
            (128, 0, 0),      # 7: Maroon
            (0, 0, 0),        # 8: Black
            (255, 255, 0),    # 9: Yellow stripe
            (0, 0, 255),      # 10: Blue stripe
            (255, 0, 0),      # 11: Red stripe
            (128, 0, 128),    # 12: Purple stripe
            (255, 165, 0),    # 13: Orange stripe
            (0, 128, 0),      # 14: Green stripe
            (128, 0, 0),      # 15: Maroon stripe
        ]
        return colors[min(ball_number, len(colors) - 1)]
    
    @override
    def reset(self) -> None:
        """Reset the environment to initial state."""
        self._setup_balls(len(self.balls))
        self.robot_arm = RobotArm()
        self.time = 0.0
        self._done = False
        self._episode_reward = 0.0
        self._last_obs = self._get_observation()
        
    @override
    def is_episode_complete(self) -> bool:
        """Check if the episode is complete."""
        return self._done
    
    @override
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation."""
        if self._last_obs is None:
            self._last_obs = self._get_observation()
        return self._last_obs
    
    def _get_observation(self) -> Dict[str, Any]:
        """Generate observation from current state."""
        # Ball positions and velocities
        ball_positions = np.array([ball.position for ball in self.balls], dtype=np.float32)
        ball_velocities = np.array([ball.velocity for ball in self.balls], dtype=np.float32)
        
        # Robot arm state
        arm_state = np.concatenate([
            self.robot_arm.joint_angles,
            self.robot_arm.end_effector_pos,
            [self.robot_arm.cue_angle, self.robot_arm.cue_power]
        ], dtype=np.float32)
        
        # Render top-down view
        image = self._render_frame()
        
        return {
            "ball_positions": ball_positions.flatten(),
            "ball_velocities": ball_velocities.flatten(), 
            "arm_state": arm_state,
            "image": image,
            "time": self.time
        }
    
    @override
    def apply_action(self, action: Dict[str, Any]) -> None:
        """Apply action to the environment."""
        if "arm_action" in action:
            # Update robot arm joint angles
            arm_action = np.array(action["arm_action"], dtype=np.float32)
            if len(arm_action) >= 6:
                self.robot_arm.joint_angles = arm_action[:6]
                
        if "cue_action" in action:
            # Execute cue strike
            cue_action = action["cue_action"]
            if isinstance(cue_action, dict):
                power = cue_action.get("power", 0.0)
                angle = cue_action.get("angle", 0.0)
                self._strike_cue_ball(power, angle)
        
        # Update physics simulation
        self._update_physics()
        self._last_obs = self._get_observation()
        
        # Check if episode is done (simplified)
        if self._all_balls_stopped():
            self._done = True
    
    def _strike_cue_ball(self, power: float, angle: float):
        """Strike the cue ball with given power and angle."""
        if len(self.balls) > 0:
            cue_ball = self.balls[0]  # First ball is cue ball
            velocity_magnitude = power * 5.0  # Scale factor
            
            cue_ball.velocity[0] = velocity_magnitude * np.cos(angle)
            cue_ball.velocity[1] = velocity_magnitude * np.sin(angle)
    
    def _update_physics(self):
        """Update physics simulation for one timestep."""
        self.time += self.dt
        
        # Update all balls
        for ball in self.balls:
            ball.update(self.dt)
            
        # Handle ball-ball collisions (simplified)
        self._handle_collisions()
        
        # Handle ball-table boundary collisions
        self._handle_boundary_collisions()
    
    def _handle_collisions(self):
        """Handle ball-to-ball collisions."""
        for i, ball1 in enumerate(self.balls):
            for j, ball2 in enumerate(self.balls[i+1:], i+1):
                distance = np.linalg.norm(ball1.position - ball2.position)
                if distance < (ball1.radius + ball2.radius):
                    # Simple elastic collision
                    self._resolve_collision(ball1, ball2)
    
    def _resolve_collision(self, ball1: Ball, ball2: Ball):
        """Resolve elastic collision between two balls."""
        # Vector from ball1 to ball2
        dx = ball2.position[0] - ball1.position[0]
        dy = ball2.position[1] - ball1.position[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance == 0:
            return
            
        # Normalize collision vector
        nx = dx / distance
        ny = dy / distance
        
        # Relative velocity
        dvx = ball2.velocity[0] - ball1.velocity[0]
        dvy = ball2.velocity[1] - ball1.velocity[1]
        
        # Relative velocity in collision normal direction
        dvn = dvx * nx + dvy * ny
        
        # Do not resolve if velocities are separating
        if dvn > 0:
            return
            
        # Collision impulse
        impulse = 2 * dvn / (ball1.mass + ball2.mass)
        
        # Update velocities
        ball1.velocity[0] += impulse * ball2.mass * nx
        ball1.velocity[1] += impulse * ball2.mass * ny
        ball2.velocity[0] -= impulse * ball1.mass * nx
        ball2.velocity[1] -= impulse * ball1.mass * ny
        
        # Separate balls to prevent overlap
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
            # Left/right boundaries
            if ball.position[0] - ball.radius < 0:
                ball.position[0] = ball.radius
                ball.velocity[0] = -ball.velocity[0] * 0.8  # Damping
            elif ball.position[0] + ball.radius > self.table.width:
                ball.position[0] = self.table.width - ball.radius
                ball.velocity[0] = -ball.velocity[0] * 0.8
                
            # Top/bottom boundaries
            if ball.position[1] - ball.radius < 0:
                ball.position[1] = ball.radius
                ball.velocity[1] = -ball.velocity[1] * 0.8
            elif ball.position[1] + ball.radius > self.table.height:
                ball.position[1] = self.table.height - ball.radius
                ball.velocity[1] = -ball.velocity[1] * 0.8
    
    def _all_balls_stopped(self) -> bool:
        """Check if all balls have stopped moving."""
        for ball in self.balls:
            if np.linalg.norm(ball.velocity) > 0.01:
                return False
        return True
    
    def _render_frame(self) -> np.ndarray:
        """Render the current frame and return as RGB array."""
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            self.clock = pygame.time.Clock()
            
        # Create surface for rendering
        surface = pygame.Surface((self.screen_width, self.screen_height))
        surface.fill((0, 100, 0))  # Green table background
        
        # Scale factor for rendering
        scale_x = self.screen_width / self.table.width
        scale_y = self.screen_height / self.table.height
        
        # Draw table boundaries
        pygame.draw.rect(surface, (139, 69, 19), 
                        (0, 0, self.screen_width, self.screen_height), 10)
        
        # Draw balls
        for ball in self.balls:
            screen_x = int(ball.position[0] * scale_x)
            screen_y = int(ball.position[1] * scale_y)
            screen_radius = int(ball.radius * min(scale_x, scale_y))
            
            pygame.draw.circle(surface, ball.color, (screen_x, screen_y), screen_radius)
            pygame.draw.circle(surface, (0, 0, 0), (screen_x, screen_y), screen_radius, 2)
            
        # Draw robot arm (simplified representation)
        arm_pos = self.robot_arm.forward_kinematics()
        arm_screen_x = int(arm_pos[0] * scale_x)
        arm_screen_y = int(arm_pos[1] * scale_y)
        
        # Draw arm base
        base_x = int(self.robot_arm.base_position[0] * scale_x)
        base_y = int(self.robot_arm.base_position[1] * scale_y)
        pygame.draw.circle(surface, (100, 100, 100), (base_x, base_y), 15)
        
        # Draw arm link
        pygame.draw.line(surface, (150, 150, 150), 
                        (base_x, base_y), (arm_screen_x, arm_screen_y), 5)
        
        # Draw end effector
        pygame.draw.circle(surface, (200, 200, 200), (arm_screen_x, arm_screen_y), 8)
        
        if self.render_mode == "human" and self.screen is not None:
            self.screen.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)
            
        # Convert to RGB array
        rgb_array = pygame.surfarray.array3d(surface)
        rgb_array = np.transpose(rgb_array, (1, 0, 2))  # Pygame uses (width, height, channels)
        
        return rgb_array.astype(np.uint8)
    
    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()


def make_example() -> Dict[str, Any]:
    """Create a random input example for the Pooltool policy."""
    return {
        "ball_positions": np.random.rand(32),  # 16 balls * 2 coordinates
        "ball_velocities": np.random.rand(32),
        "arm_state": np.random.rand(8),  # 6 joints + 2 cue params
        "image": np.random.randint(0, 256, (400, 800, 3), dtype=np.uint8),
        "time": 0.0,
        "prompt": "pot the 8 ball in the corner pocket",
    } 