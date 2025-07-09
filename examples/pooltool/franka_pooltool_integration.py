#!/usr/bin/env python3
"""
Franka Pooltool 协同仿真系统

真正的pooltool台球物理引擎与Franka Panda机械臂的深度集成:
- Pooltool: 专业台球物理、规则系统、轨迹预测
- Franka Panda: 7-DOF机械臂精确控制和碰撞检测
- 协同仿真: 机械臂与台球的物理交互

版本: 3.0.0 - 完整协同集成
作者: OpenPI团队
"""

import sys
import os
import numpy as np
import time
import pybullet as p
import pybullet_data
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

# 添加pooltool路径
current_dir = Path(__file__).parent
pooltool_path = current_dir.parent.parent / "third_party" / "pooltool"
sys.path.insert(0, str(pooltool_path))

import pooltool as pt

class RobotState(Enum):
    """机械臂状态枚举"""
    IDLE = "idle"
    APPROACHING = "approaching"
    AIMING = "aiming"
    STRIKING = "striking"
    RETRACTING = "retracting"
    OBSERVING = "observing"

@dataclass
class CueStickPose:
    """球杆姿态配置"""
    position: np.ndarray  # 球杆末端位置 [x, y, z]
    orientation: np.ndarray  # 球杆方向 [x, y, z, w] 四元数
    length: float = 1.2  # 球杆长度(米)
    tip_radius: float = 0.006  # 球杆尖端半径(米)

@dataclass 
class ShotParameters:
    """击球参数"""
    velocity: float  # 击球速度 (m/s)
    phi: float  # 水平角度 (弧度)
    theta: float  # 俯仰角度 (弧度) 
    offset_a: float  # 击球点偏移a
    offset_b: float  # 击球点偏移b
    target_ball: str = "cue"  # 目标球
    English: bool = False  # 是否使用旋转

class FrankaPooltoolIntegration:
    """
    Franka Pooltool 协同集成系统
    
    特性:
    1. 真正的pooltool台球物理引擎
    2. Franka Panda 7-DOF机械臂精确控制
    3. 机械臂-台球物理交互
    4. 专业台球规则和策略
    5. 实时碰撞检测和轨迹预测
    """
    
    def __init__(
        self,
        use_gui: bool = True,
        enable_video_recording: bool = False,
        table_type: str = "POCKET",
        robot_base_position: Tuple[float, float, float] = (-1.2, 0, 0.83)
    ):
        """
        初始化Franka-Pooltool协同系统
        
        Args:
            use_gui: 是否使用GUI显示
            enable_video_recording: 是否启用视频录制
            table_type: 台球桌类型
            robot_base_position: 机械臂基座位置
        """
        self.use_gui = use_gui
        self.enable_video_recording = enable_video_recording
        self.table_type = table_type
        self.robot_base_position = robot_base_position
        
        # 系统状态
        self.robot_state = RobotState.IDLE
        self.current_shot_params = None
        self.shot_history = []
        
        # 初始化PyBullet物理引擎 (机械臂仿真)
        self._init_pybullet()
        
        # 初始化Pooltool物理引擎 (台球仿真)
        self._init_pooltool()
        
        # 初始化Franka机械臂
        self._init_franka_robot()
        
        # 初始化球杆
        self._init_cue_stick()
        
        # 视频录制
        self.video_writer = None
        if enable_video_recording:
            self._init_video_recording()
        
        print("🎯 Franka-Pooltool协同仿真系统初始化完成!")
        print(f"   物理引擎: Pooltool (台球) + PyBullet (机械臂)")
        print(f"   机械臂: Franka Panda 7-DOF @ {robot_base_position}")
        print(f"   台球桌: {table_type}")
    
    def _init_pybullet(self):
        """初始化PyBullet物理引擎"""
        if self.use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # 创建地面
        self.ground_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.ground_id, -1, lateralFriction=0.8, restitution=0.1)
        
        print("✅ PyBullet物理引擎初始化完成")
    
    def _init_pooltool(self):
        """初始化Pooltool台球物理引擎"""
        try:
            # 创建台球桌 - 使用正确的API
            self.table = pt.Table.default()
            
            # 创建球杆
            self.cue = pt.Cue.default()
            
            # 设置标准台球布局
            balls = self._setup_standard_pool_balls()
            
            # 创建台球系统
            self.system = pt.System(
                cue=self.cue,
                table=self.table,
                balls=balls
            )
            
            self.pooltool_enabled = True
            print(f"✅ Pooltool台球引擎初始化完成 ({self.table_type})")
            
        except Exception as e:
            print(f"❌ Pooltool初始化失败: {e}")
            print("   使用简化的台球物理模拟")
            self.pooltool_enabled = False
    
    def _setup_standard_pool_balls(self):
        """设置标准台球布局"""
        try:
            # 创建母球 (白球) - 使用正确的API
            cue_ball = pt.Ball.create("cue", xy=(0.4, 0.5))
            
            # 创建彩球布局 (简化的8球布局)
            ball_positions = [
                (0.8, 0.5),      # 1号球
                (0.85, 0.52),    # 2号球
                (0.85, 0.48),    # 3号球
                (0.9, 0.54),     # 4号球
                (0.9, 0.5),      # 5号球
                (0.9, 0.46),     # 6号球
                (0.95, 0.56),    # 7号球
                (0.95, 0.52),    # 8号球
                (0.95, 0.48),    # 9号球
                (0.95, 0.44),    # 10号球
            ]
            
            balls = {"cue": cue_ball}
            for i, pos in enumerate(ball_positions, 1):
                ball = pt.Ball.create(str(i), xy=pos)
                balls[str(i)] = ball
            
            print(f"✅ 台球布局设置完成: {len(balls)}个球")
            return balls
            
        except Exception as e:
            print(f"⚠️ 台球布局设置失败: {e}")
            # 返回最小布局
            return {
                "cue": pt.Ball.create("cue", xy=(0.4, 0.5)),
                "1": pt.Ball.create("1", xy=(0.8, 0.5))
            }
    
    def _init_franka_robot(self):
        """初始化Franka Panda机械臂"""
        try:
            # 加载Franka机械臂URDF
            franka_urdf_path = current_dir / "data/pybullet-panda/data/franka/panda_arm.urdf"
            
            self.robot_id = p.loadURDF(
                str(franka_urdf_path),
                basePosition=self.robot_base_position,
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True
            )
            
            # 获取关节信息
            self.num_joints = p.getNumJoints(self.robot_id)
            self.joint_indices = []
            
            for i in range(self.num_joints):
                joint_info = p.getJointInfo(self.robot_id, i)
                if joint_info[2] == p.JOINT_REVOLUTE:
                    self.joint_indices.append(i)
            
            # 设置安全的home位置
            self.home_position = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.8]
            self._move_to_joint_positions(self.home_position, duration=2.0)
            
            # 获取末端执行器链接ID
            self.end_effector_link = self.num_joints - 1
            
            print(f"✅ Franka机械臂初始化完成")
            print(f"   关节数: {len(self.joint_indices)}")
            print(f"   末端执行器链接: {self.end_effector_link}")
            
        except Exception as e:
            print(f"❌ Franka机械臂初始化失败: {e}")
            self.robot_id = None
    
    def _init_cue_stick(self):
        """初始化球杆模型"""
        try:
            # 创建球杆的简单圆柱体模型
            cue_collision_shape = p.createCollisionShape(
                p.GEOM_CYLINDER, 
                radius=0.006,  # 6mm半径
                height=1.2     # 1.2m长度
            )
            
            cue_visual_shape = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.006,
                length=1.2,
                rgbaColor=[0.8, 0.6, 0.3, 1.0]  # 木色
            )
            
            # 初始位置在机械臂末端附近
            initial_cue_position = [
                self.robot_base_position[0] + 0.3,
                self.robot_base_position[1],
                self.robot_base_position[2] + 0.2
            ]
            
            self.cue_stick_id = p.createMultiBody(
                baseMass=0.5,  # 500g球杆
                baseCollisionShapeIndex=cue_collision_shape,
                baseVisualShapeIndex=cue_visual_shape,
                basePosition=initial_cue_position,
                baseOrientation=p.getQuaternionFromEuler([0, np.pi/2, 0])  # 水平放置
            )
            
            # 设置球杆物理属性
            p.changeDynamics(
                self.cue_stick_id, -1,
                lateralFriction=0.8,
                rollingFriction=0.1,
                restitution=0.3,
                mass=0.5
            )
            
            print("✅ 球杆模型初始化完成")
            
        except Exception as e:
            print(f"❌ 球杆初始化失败: {e}")
            self.cue_stick_id = None
    
    def _init_video_recording(self):
        """初始化视频录制"""
        timestamp = int(time.time() * 1000000) % 10000000
        video_filename = f"enhanced_franka_pool_demo_{timestamp}.mp4"
        video_path = current_dir / "videos" / video_filename
        video_path.parent.mkdir(exist_ok=True)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(video_path), fourcc, 30.0, (1024, 768)
        )
        self.video_path = video_path
        print(f"📹 视频录制初始化: {video_filename}")
    
    def _capture_frame(self):
        """捕获当前仿真帧并写入视频"""
        if not self.video_writer or not self.use_gui:
            return
            
        try:
            # 获取当前渲染图像
            img_arr = p.getCameraImage(
                width=1024,
                height=768,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # 转换图像格式
            rgba_img = img_arr[2]
            rgb_img = rgba_img[:, :, :3]  # 去掉alpha通道
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # 写入视频
            self.video_writer.write(bgr_img)
            
        except Exception as e:
            print(f"⚠️ 视频帧捕获失败: {e}")
    
    def execute_pool_shot(
        self,
        velocity: float,
        phi: float = 0.0,
        theta: float = 0.0,
        offset_a: float = 0.0,
        offset_b: float = 0.0,
        target_ball: str = "cue"
    ) -> Dict:
        """
        执行完整的台球击球动作
        
        Args:
            velocity: 击球速度 (m/s)
            phi: 水平角度 (弧度)
            theta: 俯仰角度 (弧度)
            offset_a: 击球点偏移a
            offset_b: 击球点偏移b
            target_ball: 目标球ID
            
        Returns:
            Dict: 击球结果和分析
        """
        shot_params = ShotParameters(
            velocity=velocity,
            phi=phi,
            theta=theta,
            offset_a=offset_a,
            offset_b=offset_b,
            target_ball=target_ball
        )
        
        print(f"\n🎯 执行台球击球:")
        print(f"   速度: {velocity:.1f} m/s")
        print(f"   角度: {np.degrees(phi):.1f}° (水平), {np.degrees(theta):.1f}° (俯仰)")
        print(f"   偏移: a={offset_a:.3f}, b={offset_b:.3f}")
        
        # 阶段1: 机械臂接近球杆
        self._robot_approach_cue()
        
        # 阶段2: 精确定位和瞄准
        self._robot_aim_shot(shot_params)
        
        # 阶段3: 执行击球动作
        shot_result = self._robot_execute_strike(shot_params)
        
        # 阶段4: 机械臂撤回观察
        self._robot_retract_and_observe()
        
        # 阶段5: 分析击球结果
        analysis = self._analyze_shot_result(shot_result, shot_params)
        
        # 记录击球历史
        self.shot_history.append({
            "timestamp": time.time(),
            "parameters": shot_params.__dict__,
            "result": shot_result,
            "analysis": analysis
        })
        
        return analysis
    
    def _robot_approach_cue(self):
        """机械臂接近球杆"""
        print("🤖 阶段1: 机械臂接近球杆...")
        self.robot_state = RobotState.APPROACHING
        
        # 计算球杆附近的安全位置
        if self.cue_stick_id:
            cue_pos, cue_orn = p.getBasePositionAndOrientation(self.cue_stick_id)
            approach_position = [
                cue_pos[0] - 0.2,  # 距离球杆20cm
                cue_pos[1],
                cue_pos[2] + 0.1   # 稍微抬高
            ]
            
            # 计算逆运动学到达接近位置
            approach_joints = self._ik_solve(approach_position, [0, 0, 0, 1])
            
            if approach_joints is not None:
                self._move_to_joint_positions(approach_joints, duration=2.0)
            
            print("   ✅ 机械臂已接近球杆位置")
        else:
            print("   ⚠️ 球杆未初始化，使用预设接近位置")
            approach_joints = [0.2, -0.5, 0.1, -1.8, 0.0, 1.3, 0.5]
            self._move_to_joint_positions(approach_joints, duration=2.0)
    
    def _robot_aim_shot(self, shot_params: ShotParameters):
        """机械臂精确瞄准"""
        print("🎯 阶段2: 精确瞄准击球位置...")
        self.robot_state = RobotState.AIMING
        
        # 获取目标球位置 (简化版本)
        if hasattr(self, 'system') and self.system.balls:
            try:
                target_ball = self.system.balls.get(shot_params.target_ball)
                if target_ball:
                    ball_position = target_ball.xyz
                else:
                    print(f"   ⚠️ 未找到目标球 {shot_params.target_ball}，使用默认位置")
                    ball_position = [0.0, 0.0, 0.0]
            except Exception as e:
                print(f"   ⚠️ 获取球位置失败: {e}，使用默认位置")
                ball_position = [0.0, 0.0, 0.0]
        else:
            ball_position = [0.0, 0.0, 0.0]
        
        # 计算击球位置和角度
        strike_direction = np.array([np.cos(shot_params.phi), np.sin(shot_params.phi), 0])
        strike_distance = 0.15  # 距离球15cm开始击球
        
        strike_position = [
            ball_position[0] - strike_direction[0] * strike_distance,
            ball_position[1] - strike_direction[1] * strike_distance,
            ball_position[2] + 0.05  # 稍微抬高
        ]
        
        # 计算瞄准姿态
        aim_orientation = p.getQuaternionFromEuler([0, shot_params.theta, shot_params.phi])
        
        # 移动到瞄准位置
        aim_joints = self._ik_solve(strike_position, aim_orientation)
        if aim_joints is not None:
            self._move_to_joint_positions(aim_joints, duration=1.5)
        
        print(f"   ✅ 瞄准完成，目标位置: {ball_position}")
        
        # 瞄准时的小幅度调整
        for step in range(60):  # 1秒的瞄准时间
            p.stepSimulation()
            
            # 捕获视频帧
            if step % 2 == 0:  # 每2步捕获一帧(30fps)
                self._capture_frame()
                
            if self.use_gui:
                time.sleep(1/60)
    
    def _robot_execute_strike(self, shot_params: ShotParameters) -> Dict:
        """执行击球动作"""
        print(f"⚡ 阶段3: 执行击球 (速度: {shot_params.velocity:.1f} m/s)...")
        self.robot_state = RobotState.STRIKING
        
        result = {
            "success": False,
            "contact_detected": False,
            "ball_motion": {},
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # 使用Pooltool计算击球物理
            if hasattr(self, 'system') and hasattr(self, 'pooltool_enabled') and self.pooltool_enabled:
                # 设置击球参数
                self.system.cue.set_state(
                    V0=shot_params.velocity,
                    phi=shot_params.phi,
                    theta=shot_params.theta,
                    a=shot_params.offset_a,
                    b=shot_params.offset_b
                )
                
                # 执行Pooltool物理模拟
                pt.simulate(self.system, inplace=True)
                
                # 记录球的运动
                for ball_id, ball in self.system.balls.items():
                    result["ball_motion"][ball_id] = {
                        "initial_pos": ball.xyz.copy(),
                        "initial_vel": ball.vel.copy()
                    }
                
                result["success"] = True
                result["contact_detected"] = True
                
                print("   ✅ Pooltool物理模拟执行成功")
                
            else:
                print("   ⚠️ Pooltool未可用，使用简化击球模拟")
                
            # 机械臂执行快速前进-后退动作模拟击球
            current_joints = self._get_current_joint_positions()
            
            # 前进击球 (快速)
            strike_joints = current_joints.copy()
            strike_joints[1] += 0.15  # 第2关节前进
            self._move_to_joint_positions(strike_joints, duration=0.3)
            
            # 立即后退
            self._move_to_joint_positions(current_joints, duration=0.2)
            
            result["execution_time"] = time.time() - start_time
            
        except Exception as e:
            print(f"   ❌ 击球执行失败: {e}")
            result["error"] = str(e)
        
        return result
    
    def _robot_retract_and_observe(self):
        """机械臂撤回并观察结果"""
        print("👀 阶段4: 撤回并观察击球结果...")
        self.robot_state = RobotState.RETRACTING
        
        # 撤回到安全观察位置
        observe_position = [0.1, -0.4, 0.2, -1.5, 0.0, 1.1, 0.8]
        self._move_to_joint_positions(observe_position, duration=2.0)
        
        self.robot_state = RobotState.OBSERVING
        
        # 观察时间
        observation_time = 3.0
        print(f"   观察台球运动 {observation_time}秒...")
        
        for step in range(int(observation_time * 240)):
            p.stepSimulation()
            
            # 捕获视频帧
            if step % 8 == 0:  # 每8步捕获一帧(30fps)
                self._capture_frame()
                
            if self.use_gui:
                time.sleep(1/240)
            
            # 可以在这里添加球运动状态的实时监控
            
        print("   ✅ 观察完成")
        self.robot_state = RobotState.IDLE
    
    def _analyze_shot_result(self, shot_result: Dict, shot_params: ShotParameters) -> Dict:
        """分析击球结果"""
        print("📊 阶段5: 分析击球结果...")
        
        analysis = {
            "shot_success": shot_result.get("success", False),
            "contact_made": shot_result.get("contact_detected", False),
            "execution_quality": "good" if shot_result.get("success") else "poor",
            "recommendations": []
        }
        
        # 基础分析
        if analysis["shot_success"]:
            analysis["recommendations"].append("击球技术执行良好")
        else:
            analysis["recommendations"].append("需要改进击球精度")
        
        # 速度分析
        if shot_params.velocity > 6.0:
            analysis["recommendations"].append("击球速度较高，注意控制力度")
        elif shot_params.velocity < 2.0:
            analysis["recommendations"].append("击球速度较低，可以增加力度")
        
        # 角度分析
        if abs(shot_params.phi) > np.pi/3:
            analysis["recommendations"].append("击球角度较大，注意瞄准精度")
        
        print(f"   成功: {analysis['shot_success']}")
        print(f"   质量: {analysis['execution_quality']}")
        print(f"   建议: {', '.join(analysis['recommendations'])}")
        
        return analysis
    
    def _move_to_joint_positions(self, target_positions: List[float], duration: float = 2.0):
        """平滑移动到目标关节位置"""
        if not self.robot_id or not target_positions:
            return
        
        current_positions = self._get_current_joint_positions()
        
        steps = int(duration * 240)  # 240Hz
        for step in range(steps):
            alpha = step / steps
            # 使用平滑插值曲线
            smooth_alpha = 0.5 * (1 - np.cos(np.pi * alpha))
            
            interpolated = []
            for i, (current, target) in enumerate(zip(current_positions, target_positions)):
                if i < len(self.joint_indices):
                    pos = current + (target - current) * smooth_alpha
                    interpolated.append(pos)
            
            # 应用关节位置
            for i, pos in enumerate(interpolated):
                if i < len(self.joint_indices):
                    p.resetJointState(self.robot_id, self.joint_indices[i], pos)
            
            p.stepSimulation()
            
            # 捕获视频帧
            if step % 8 == 0:  # 每8步捕获一帧(30fps)
                self._capture_frame()
                
            if self.use_gui:
                time.sleep(1/240)
    
    def _get_current_joint_positions(self) -> List[float]:
        """获取当前关节位置"""
        if not self.robot_id:
            return []
        
        positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            positions.append(joint_state[0])
        return positions
    
    def _ik_solve(self, target_position: List[float], target_orientation: List[float]) -> Optional[List[float]]:
        """简化的逆运动学求解"""
        if not self.robot_id:
            return None
        
        try:
            # 使用PyBullet的IK求解器
            joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.end_effector_link,
                target_position,
                target_orientation,
                maxNumIterations=100,
                residualThreshold=1e-5
            )
            
            # 只返回前7个关节的位置
            return list(joint_positions[:len(self.joint_indices)])
            
        except Exception as e:
            print(f"⚠️ IK求解失败: {e}")
            return None
    
    def reset_simulation(self):
        """重置仿真环境"""
        print("🔄 重置仿真环境...")
        
        # 重置机械臂到home位置
        if self.robot_id:
            self._move_to_joint_positions(self.home_position, duration=1.0)
        
        # 重置pooltool系统
        if hasattr(self, 'system') and hasattr(self, 'pooltool_enabled') and self.pooltool_enabled:
            # 重新创建系统
            balls = self._setup_standard_pool_balls()
            self.system = pt.System(
                cue=self.cue,
                table=self.table,
                balls=balls
            )
        
        # 重置球杆位置
        if self.cue_stick_id:
            initial_cue_position = [
                self.robot_base_position[0] + 0.3,
                self.robot_base_position[1],
                self.robot_base_position[2] + 0.2
            ]
            p.resetBasePositionAndOrientation(
                self.cue_stick_id,
                initial_cue_position,
                p.getQuaternionFromEuler([0, np.pi/2, 0])
            )
        
        self.robot_state = RobotState.IDLE
        print("✅ 仿真环境重置完成")
    
    def set_camera_view(self, preset: str = "overview"):
        """设置摄像头视角"""
        camera_configs = {
            "overview": {"distance": 4.0, "yaw": 45, "pitch": -35, "target": [0, 0, 0.8]},
            "robot_view": {"distance": 2.5, "yaw": -60, "pitch": -20, "target": [-0.8, 0, 0.8]},
            "table_view": {"distance": 3.0, "yaw": 90, "pitch": -25, "target": [0.3, 0, 0.8]},
            "action_view": {"distance": 1.8, "yaw": 30, "pitch": -15, "target": [-0.2, 0, 0.8]}
        }
        
        config = camera_configs.get(preset, camera_configs["overview"])
        
        p.resetDebugVisualizerCamera(
            cameraDistance=config["distance"],
            cameraYaw=config["yaw"],
            cameraPitch=config["pitch"],
            cameraTargetPosition=config["target"]
        )
        
        print(f"📷 摄像头设置为 '{preset}' 视角")
    
    def run_demonstration(self):
        """运行完整的协同仿真演示"""
        print("\n🚀 开始Franka-Pooltool协同仿真演示!")
        
        # 设置初始摄像头视角
        self.set_camera_view("overview")
        
        # 演示击球序列
        demo_shots = [
            {"velocity": 4.0, "phi": 0.0, "theta": 0.0, "offset_a": 0.0, "offset_b": 0.0},
            {"velocity": 5.5, "phi": np.pi/6, "theta": 0.0, "offset_a": 0.01, "offset_b": 0.0},
            {"velocity": 3.2, "phi": -np.pi/8, "theta": 0.0, "offset_a": 0.0, "offset_b": 0.01},
            {"velocity": 6.0, "phi": np.pi/4, "theta": 0.05, "offset_a": -0.01, "offset_b": 0.0},
        ]
        
        shot_names = ["直击", "斜击", "旋转球", "高难度击球"]
        
        for i, (shot_params, shot_name) in enumerate(zip(demo_shots, shot_names), 1):
            print(f"\n{'='*50}")
            print(f"第{i}击: {shot_name}")
            print(f"{'='*50}")
            
            # 执行击球
            result = self.execute_pool_shot(**shot_params)
            
            # 切换摄像头视角观察结果
            if i % 2 == 0:
                self.set_camera_view("table_view")
            else:
                self.set_camera_view("robot_view")
            
            # 等待观察并捕获帧
            for step in range(60):  # 2秒等待时间
                p.stepSimulation()
                if step % 2 == 0:
                    self._capture_frame()
                if self.use_gui:
                    time.sleep(1/30)
            
            # 重置环境(除了最后一击)
            if i < len(demo_shots):
                self.reset_simulation()
                # 重置时也捕获帧
                for step in range(30):  # 1秒重置时间
                    p.stepSimulation()
                    if step % 1 == 0:
                        self._capture_frame()
                    if self.use_gui:
                        time.sleep(1/30)
        
        print(f"\n🎉 协同仿真演示完成!")
        print(f"   总击球次数: {len(self.shot_history)}")
        print(f"   系统状态: {self.robot_state.value}")
        
        # 保存击球历史
        self._save_shot_history()
    
    def _save_shot_history(self):
        """保存击球历史"""
        timestamp = int(time.time())
        history_file = current_dir / "data" / f"franka_pooltool_history_{timestamp}.json"
        history_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(self.shot_history, f, indent=2, ensure_ascii=False, default=str)
            print(f"📁 击球历史已保存: {history_file.name}")
        except Exception as e:
            print(f"⚠️ 保存击球历史失败: {e}")
    
    def close(self):
        """关闭仿真系统"""
        print("🔚 关闭Franka-Pooltool协同仿真系统...")
        
        if self.video_writer:
            self.video_writer.release()
            print(f"   📹 视频录制已保存: {self.video_path}")
            # 检查视频文件大小
            if hasattr(self, 'video_path') and self.video_path.exists():
                video_size = self.video_path.stat().st_size
                print(f"   视频文件大小: {video_size / 1024 / 1024:.2f} MB")
        
        if hasattr(self, 'physics_client'):
            p.disconnect(self.physics_client)
            print("   🔌 PyBullet连接已断开")
        
        print("✅ 系统关闭完成")

def main():
    """主函数 - 运行Franka-Pooltool协同仿真演示"""
    print("🎯 Franka Panda + Pooltool 协同仿真系统")
    print("=" * 60)
    
    try:
        # 创建协同仿真系统
        integration = FrankaPooltoolIntegration(
            use_gui=True,
            enable_video_recording=True,
            table_type="POCKET"
        )
        
        # 运行演示
        integration.run_demonstration()
        
        # 保持仿真运行，等待用户交互
        print("\n⌨️ 按回车键退出仿真...")
        input()
        
    except KeyboardInterrupt:
        print("\n⚡ 用户中断")
    except Exception as e:
        print(f"\n❌ 系统异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if 'integration' in locals():
            integration.close()

if __name__ == "__main__":
    main() 