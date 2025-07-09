#!/usr/bin/env python3
"""
真正的Pooltool物理桥接器

这个模块实现了pooltool专业台球物理引擎与PyBullet机械臂仿真的完整集成。
- Pooltool: 专业台球物理仿真、碰撞检测、轨迹计算
- PyBullet: 机械臂渲染和控制

作者: OpenPI团队  
版本: 2.0.0 - 真正的pooltool集成版本
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# 添加third_party/pooltool到路径
current_dir = Path(__file__).parent
pooltool_path = current_dir.parent.parent / "third_party" / "pooltool"
sys.path.insert(0, str(pooltool_path))

# 导入pooltool
import pooltool as pt
import pybullet as p

@dataclass
class BallState:
    """球状态数据类"""
    ball_id: str
    position: np.ndarray
    velocity: np.ndarray 
    angular_velocity: np.ndarray
    active: bool = True
    pocketed: bool = False

@dataclass
class CueAction:
    """击球动作参数"""
    velocity: float = 5.0      # 击球速度 (m/s)
    phi: float = 0.0           # 水平角度 (弧度)
    theta: float = 0.0         # 垂直角度 (弧度) 
    offset_a: float = 0.0      # 击球点水平偏移
    offset_b: float = 0.0      # 击球点垂直偏移

class PoolToolPhysicsEngine:
    """
    Pooltool专业台球物理引擎
    
    这是核心物理引擎，负责：
    - 专业台球物理仿真
    - 球球碰撞、球桌碰撞
    - 摩擦、旋转、滚动效应
    - 进袋检测和轨迹计算
    """
    
    def __init__(self, table_type: str = "POCKET"):
        """
        初始化Pooltool物理引擎
        
        Args:
            table_type: 台球桌类型 ("POCKET", "SNOOKER", "CAROM")
        """
        self.table_type = getattr(pt.TableType, table_type, pt.TableType.POCKET)
        self.system = None
        self.table = None
        self.cue = None
        self.shot_history = []
        
        # 初始化系统
        self._setup_system()
        
    def _setup_system(self):
        """设置pooltool台球系统"""
        try:
            # 创建专业台球桌
            self.table = pt.Table.default(table_type=self.table_type)
            
            # 创建击球杆
            self.cue = pt.Cue.default()
            
            # 创建标准球堆 (15球 + 主球)
            balls = self._create_standard_ball_rack()
            
            # 创建完整系统
            self.system = pt.System(
                table=self.table,
                cue=self.cue,
                balls=balls
            )
            
            print(f"✅ Pooltool专业台球系统创建成功")
            print(f"   - 台球桌类型: {self.table_type}")
            print(f"   - 球数量: {len(balls)}")
            print(f"   - 桌面尺寸: {self.table.l} x {self.table.w}")
            
        except Exception as e:
            print(f"❌ Pooltool系统创建失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_standard_ball_rack(self) -> Dict[str, pt.Ball]:
        """创建标准台球球堆"""
        balls = {}
        
        # 主球 (白球)
        cue_ball = pt.Ball.create("cue")
        # 使用正确的API设置位置
        cue_ball.state.rvw[0] = np.array([-self.table.l/4, 0, cue_ball.params.R])  # 左侧1/4位置
        cue_ball.state.rvw[1] = np.array([0, 0, 0])  # 静止速度
        cue_ball.state.rvw[2] = np.array([0, 0, 0])  # 无角速度
        cue_ball.state.s = 0  # stationary状态
        balls["cue"] = cue_ball
        
        # 目标球堆 (1-15号球)
        rack_center = np.array([self.table.l/4, 0, 0])  # 右侧1/4位置
        ball_positions = self._generate_rack_positions(rack_center)
        
        for i, pos in enumerate(ball_positions, 1):
            ball_id = str(i)
            ball = pt.Ball.create(ball_id)
            # 设置球的位置
            ball.state.rvw[0] = pos
            ball.state.rvw[1] = np.array([0, 0, 0])  # 静止速度
            ball.state.rvw[2] = np.array([0, 0, 0])  # 无角速度
            ball.state.s = 0  # stationary状态
            balls[ball_id] = ball
        
        return balls
    
    def _generate_rack_positions(self, center: np.ndarray, spacing: float = 0.057) -> List[np.ndarray]:
        """生成标准三角形球堆位置"""
        positions = []
        ball_radius = 0.02835  # 标准台球半径
        
        # 5行三角形排列
        rows = [1, 2, 3, 4, 5]
        y_start = -2 * spacing  # 从中心向两侧分布
        
        for row_idx, balls_in_row in enumerate(rows):
            x_offset = row_idx * spacing * np.sqrt(3) / 2
            y_positions = np.linspace(
                -(balls_in_row - 1) * spacing / 2,
                (balls_in_row - 1) * spacing / 2, 
                balls_in_row
            )
            
            for y_offset in y_positions:
                pos = center + np.array([x_offset, y_offset, ball_radius])
                positions.append(pos)
                if len(positions) >= 15:  # 最多15个目标球
                    return positions
        
        return positions
    
    def execute_shot(self, cue_action: CueAction) -> pt.System:
        """
        执行击球并返回仿真结果
        
        Args:
            cue_action: 击球参数
            
        Returns:
            仿真后的系统状态
        """
        # 设置击球参数 - 使用正确的API
        self.cue.set_state(
            V0=cue_action.velocity,
            phi=np.degrees(cue_action.phi),  # pooltool使用度数
            theta=np.degrees(cue_action.theta),
            a=cue_action.offset_a,
            b=cue_action.offset_b
        )
        
        print(f"🎱 执行击球: 速度={cue_action.velocity:.1f}m/s, 角度={np.degrees(cue_action.phi):.1f}°")
        
        # 运行pooltool专业物理仿真
        try:
            simulated_system = pt.simulate(
                self.system,
                inplace=False,
                continuous=True,  # 生成连续轨迹
                dt=0.008          # 8ms时间步长，专业精度
            )
            
            # 保存击球历史
            self.shot_history.append({
                'shot_params': cue_action,
                'initial_state': self.system,
                'result_state': simulated_system,
                'timestamp': time.time()
            })
            
            # 更新当前系统状态
            self.system = simulated_system
            
            print(f"✅ 击球仿真完成，轨迹长度: {len(simulated_system.balls['cue'].history)} 帧")
            
            return simulated_system
            
        except Exception as e:
            print(f"❌ 击球仿真失败: {e}")
            import traceback
            traceback.print_exc()
            return self.system
    
    def get_ball_states(self) -> Dict[str, BallState]:
        """获取所有球的当前状态"""
        states = {}
        
        if not self.system or not self.system.balls:
            return states
        
        for ball_id, ball in self.system.balls.items():
            try:
                # 获取球的当前状态
                position = ball.state.rvw if hasattr(ball.state, 'rvw') else np.array([0, 0, 0])
                velocity = ball.state.s[:3] if hasattr(ball.state, 's') and len(ball.state.s) >= 3 else np.array([0, 0, 0])
                angular_vel = ball.state.s[3:6] if hasattr(ball.state, 's') and len(ball.state.s) >= 6 else np.array([0, 0, 0])
                
                # 检查球是否被击入袋中
                pocketed = getattr(ball.state, 'pocketed', False) if hasattr(ball, 'state') else False
                
                states[ball_id] = BallState(
                    ball_id=ball_id,
                    position=np.array(position),
                    velocity=np.array(velocity),
                    angular_velocity=np.array(angular_vel),
                    active=not pocketed,
                    pocketed=pocketed
                )
                
            except Exception as e:
                print(f"Warning: 无法获取球 {ball_id} 的状态: {e}")
                # 提供默认状态
                states[ball_id] = BallState(
                    ball_id=ball_id,
                    position=np.array([0, 0, 0]),
                    velocity=np.array([0, 0, 0]),
                    angular_velocity=np.array([0, 0, 0]),
                    active=True,
                    pocketed=False
                )
        
        return states
    
    def get_pocketed_balls(self) -> List[str]:
        """获取已进袋的球"""
        pocketed = []
        states = self.get_ball_states()
        
        for ball_id, state in states.items():
            if state.pocketed:
                pocketed.append(ball_id)
        
        return pocketed
    
    def detect_collisions(self) -> List[Tuple[str, str]]:
        """检测球球碰撞"""
        collisions = []
        
        # pooltool的事件系统会自动检测碰撞
        if hasattr(self.system, 'events') and self.system.events:
            for event in self.system.events:
                if event.event_type == pt.EventType.BALL_BALL:
                    ball1_id = event.ball_a.id if hasattr(event, 'ball_a') else 'unknown'
                    ball2_id = event.ball_b.id if hasattr(event, 'ball_b') else 'unknown'
                    collisions.append((ball1_id, ball2_id))
        
        return collisions
    
    def reset_to_break_formation(self):
        """重置为开球状态"""
        balls = self._create_standard_ball_rack()
        self.system = pt.System(
            table=self.table,
            cue=self.cue, 
            balls=balls
        )
        print("🔄 重置为开球状态")
    
    def save_state(self, filepath: str):
        """保存当前系统状态"""
        try:
            self.system.save(filepath)
            print(f"💾 系统状态已保存: {filepath}")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def load_state(self, filepath: str):
        """加载系统状态"""
        try:
            self.system = pt.System.load(filepath)
            print(f"📂 系统状态已加载: {filepath}")
        except Exception as e:
            print(f"❌ 加载失败: {e}")

class PyBulletRobotRenderer:
    """
    PyBullet机械臂渲染器
    
    专门负责Franka机械臂的3D渲染和控制，
    与pooltool物理引擎协同工作
    """
    
    def __init__(self, physics_client_id: int, robot_position: Tuple[float, float, float] = (-1.2, 0, 0.83)):
        """
        初始化机械臂渲染器
        
        Args:
            physics_client_id: PyBullet客户端ID
            robot_position: 机械臂基座位置
        """
        self.physics_client = physics_client_id
        self.robot_position = robot_position
        self.robot_id = None
        self.joint_indices = []
        
        # 加载Franka机械臂
        self._load_franka_robot()
    
    def _load_franka_robot(self):
        """加载Franka Panda机械臂模型"""
        try:
            # Franka URDF路径
            franka_urdf_path = current_dir / "data/pybullet-panda/data/franka/panda_arm.urdf"
            
            if not franka_urdf_path.exists():
                raise FileNotFoundError(f"Franka URDF不存在: {franka_urdf_path}")
            
            # 加载机械臂
            self.robot_id = p.loadURDF(
                str(franka_urdf_path),
                basePosition=self.robot_position,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                physicsClientId=self.physics_client
            )
            
            # 获取关节信息
            num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
            
            for i in range(num_joints):
                joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
                if joint_info[2] == p.JOINT_REVOLUTE:  # 只有旋转关节
                    self.joint_indices.append(i)
            
            # 设置安全初始位置
            home_position = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.8]
            for i, joint_idx in enumerate(self.joint_indices[:len(home_position)]):
                p.resetJointState(
                    self.robot_id, joint_idx, home_position[i],
                    physicsClientId=self.physics_client
                )
            
            print(f"✅ Franka机械臂渲染器初始化成功")
            print(f"   - 位置: {self.robot_position}")
            print(f"   - 可控关节数: {len(self.joint_indices)}")
            
        except Exception as e:
            print(f"❌ Franka机械臂加载失败: {e}")
            self.robot_id = None
    
    def move_to_position(self, joint_positions: List[float], duration: float = 2.0):
        """平滑移动到目标关节位置"""
        if not self.robot_id or not joint_positions:
            return
        
        # 获取当前关节位置
        current_positions = []
        for joint_idx in self.joint_indices[:len(joint_positions)]:
            joint_state = p.getJointState(
                self.robot_id, joint_idx, 
                physicsClientId=self.physics_client
            )
            current_positions.append(joint_state[0])
        
        # 平滑插值移动
        steps = int(duration * 240)  # 240Hz
        for step in range(steps):
            alpha = step / steps
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))  # 平滑曲线
            
            interpolated = []
            for current, target in zip(current_positions, joint_positions):
                pos = current + (target - current) * smooth_alpha
                interpolated.append(pos)
            
            # 应用关节位置
            for joint_idx, pos in zip(self.joint_indices[:len(interpolated)], interpolated):
                p.resetJointState(
                    self.robot_id, joint_idx, pos,
                    physicsClientId=self.physics_client
                )
            
            p.stepSimulation(physicsClientId=self.physics_client)
            time.sleep(1/240)

class TruePooltoolBridge:
    """
    真正的Pooltool物理桥接器
    
    集成pooltool专业台球物理引擎与PyBullet机械臂渲染，
    实现完整的台球机器人仿真系统
    """
    
    def __init__(
        self,
        physics_client_id: int,
        table_type: str = "POCKET",
        robot_position: Tuple[float, float, float] = (-1.2, 0, 0.83),
        enable_3d_viz: bool = True
    ):
        """
        初始化真正的pooltool桥接器
        
        Args:
            physics_client_id: PyBullet客户端ID
            table_type: 台球桌类型
            robot_position: 机械臂位置
            enable_3d_viz: 是否启用3D可视化
        """
        self.physics_client = physics_client_id
        self.enable_3d_viz = enable_3d_viz
        
        print("🚀 初始化真正的Pooltool集成系统...")
        
        # 初始化pooltool物理引擎 (主引擎)
        self.pool_engine = PoolToolPhysicsEngine(table_type)
        
        # 初始化PyBullet机械臂渲染器 (辅助渲染)
        self.robot_renderer = PyBulletRobotRenderer(physics_client_id, robot_position)
        
        print("✅ 真正的Pooltool集成系统初始化完成!")
        print("   - 物理引擎: Pooltool专业台球物理")
        print("   - 机械臂渲染: PyBullet Franka Panda")
        print("   - 集成状态: 完全协同工作")
    
    def execute_shot(
        self, 
        velocity: float = 5.0,
        angle_deg: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0
    ) -> Dict[str, Any]:
        """
        执行击球动作
        
        Args:
            velocity: 击球速度 (m/s)
            angle_deg: 击球角度 (度)
            offset_x: 击球点X偏移
            offset_y: 击球点Y偏移
            
        Returns:
            击球结果信息
        """
        # 创建击球动作
        cue_action = CueAction(
            velocity=velocity,
            phi=np.radians(angle_deg),
            theta=0.0,
            offset_a=offset_x,
            offset_b=offset_y
        )
        
        # 执行击球 (使用pooltool专业物理)
        result_system = self.pool_engine.execute_shot(cue_action)
        
        # 获取结果
        ball_states = self.pool_engine.get_ball_states()
        collisions = self.pool_engine.detect_collisions()
        pocketed = self.pool_engine.get_pocketed_balls()
        
        return {
            'ball_states': ball_states,
            'collisions': collisions,
            'pocketed_balls': pocketed,
            'system': result_system,
            'shot_params': cue_action
        }
    
    def get_ball_states(self) -> Dict[str, BallState]:
        """获取所有球状态"""
        return self.pool_engine.get_ball_states()
    
    def move_robot(self, joint_positions: List[float], duration: float = 2.0):
        """移动机械臂到指定位置"""
        self.robot_renderer.move_to_position(joint_positions, duration)
    
    def reset_table(self):
        """重置台球桌为开球状态"""
        self.pool_engine.reset_to_break_formation()
    
    def show_3d_visualization(self):
        """显示pooltool 3D可视化"""
        if self.enable_3d_viz:
            try:
                # 使用pooltool的专业3D可视化
                pt.show(self.pool_engine.system)
            except Exception as e:
                print(f"⚠️ 3D可视化启动失败: {e}")
                print("这可能是由于显示环境限制")
    
    def save_simulation(self, filepath: str):
        """保存完整仿真状态"""
        self.pool_engine.save_state(filepath)
    
    def get_shot_history(self) -> List[Dict[str, Any]]:
        """获取击球历史"""
        return self.pool_engine.shot_history
    
    def step_simulation(self, dt: float = 1/240):
        """推进仿真一步"""
        # PyBullet步进 (仅用于机械臂渲染)
        p.stepSimulation(physicsClientId=self.physics_client)
        
        # pooltool物理由内部自动处理，不需要手动步进

# 向后兼容的别名
PhysicsBridge = TruePooltoolBridge 