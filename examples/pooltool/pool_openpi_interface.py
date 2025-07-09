"""
Pool OpenPI Interface - 台球环境的OpenPI策略接口

整合台球机械臂环境与OpenPI推理系统，实现数据预处理、
策略推理和动作执行的完整管道。

作者: OpenPI团队
版本: 1.0.0
"""

import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import time
import threading
import queue
from abc import ABC, abstractmethod

from openpi_client import websocket_client_policy, image_tools
from pool_robot_env import PoolRobotEnvironment
from physics_bridge import PhysicsBridge


@dataclass
class PoolObservation:
    """台球环境观测数据结构"""
    # 图像观测
    overhead_image: np.ndarray      # 俯视图 (H, W, 3)
    wrist_image: np.ndarray         # 手腕相机图 (H, W, 3)
    
    # 机械臂状态
    joint_positions: np.ndarray     # 关节位置 (7,)
    joint_velocities: np.ndarray    # 关节速度 (7,)
    ee_position: np.ndarray         # 末端执行器位置 (3,)
    ee_orientation: np.ndarray      # 末端执行器方向 (4,) - 四元数
    
    # 台球状态
    ball_positions: Dict[str, np.ndarray]  # 球位置
    ball_velocities: Dict[str, np.ndarray] # 球速度
    
    # 任务信息
    task_description: str           # 任务描述
    timestamp: float               # 时间戳


@dataclass
class PoolAction:
    """台球环境动作数据结构"""
    # 机械臂动作
    joint_targets: np.ndarray      # 目标关节位置 (7,)
    gripper_action: float          # 夹爪动作 (-1 到 1)
    
    # 可选的高级动作
    cue_stick_pose: Optional[np.ndarray] = None  # 球杆姿态 (7,) [pos, quat]
    strike_force: Optional[float] = None         # 击球力度
    strike_direction: Optional[np.ndarray] = None # 击球方向


class PoolTaskDefinition:
    """台球任务定义"""
    
    def __init__(self, task_type: str, **kwargs):
        self.task_type = task_type
        self.params = kwargs
    
    def get_description(self) -> str:
        """获取任务描述"""
        if self.task_type == "pot_ball":
            target_ball = self.params.get("target_ball", "any")
            target_pocket = self.params.get("target_pocket", "any")
            return f"Pot the {target_ball} ball into the {target_pocket} pocket"
        
        elif self.task_type == "break_shot":
            return "Perform a break shot to start the game"
        
        elif self.task_type == "position_cue":
            return "Position the cue stick for the next shot"
        
        elif self.task_type == "clear_table":
            return "Clear all balls from the table"
        
        else:
            return f"Complete the {self.task_type} task"
    
    def get_success_criteria(self) -> Dict[str, Any]:
        """获取成功标准"""
        if self.task_type == "pot_ball":
            return {
                "balls_potted": self.params.get("required_balls", 1),
                "target_ball": self.params.get("target_ball", None),
                "avoid_scratch": True
            }
        
        elif self.task_type == "break_shot":
            return {
                "balls_contacted": True,
                "min_balls_moved": 4,
                "avoid_scratch": True
            }
        
        return {"task_completed": True}


class ObservationProcessor:
    """观测数据处理器"""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
    
    def process_observation(self, raw_obs: Dict[str, Any]) -> PoolObservation:
        """处理原始观测数据"""
        # 处理图像
        overhead_img = self._process_image(raw_obs.get("overhead_camera", None))
        wrist_img = self._process_image(raw_obs.get("wrist_camera", None))
        
        # 处理机械臂状态
        joint_pos = np.array(raw_obs.get("joint_positions", [0]*7))
        joint_vel = np.array(raw_obs.get("joint_velocities", [0]*7))
        ee_pos = np.array(raw_obs.get("ee_position", [0, 0, 0]))
        ee_orn = np.array(raw_obs.get("ee_orientation", [0, 0, 0, 1]))
        
        # 处理台球状态
        ball_states = raw_obs.get("pool_balls", {})
        ball_positions = {}
        ball_velocities = {}
        
        for ball_id, state in ball_states.items():
            ball_positions[ball_id] = np.array(state.get("position", [0, 0, 0]))
            ball_velocities[ball_id] = np.array(state.get("velocity", [0, 0, 0]))
        
        return PoolObservation(
            overhead_image=overhead_img,
            wrist_image=wrist_img,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            ee_position=ee_pos,
            ee_orientation=ee_orn,
            ball_positions=ball_positions,
            ball_velocities=ball_velocities,
            task_description=raw_obs.get("task_description", ""),
            timestamp=time.time()
        )
    
    def _process_image(self, image: Optional[np.ndarray]) -> np.ndarray:
        """处理单张图像"""
        if image is None:
            # 返回黑色图像
            return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        
        # 调整尺寸
        if image.shape[:2] != (self.image_size, self.image_size):
            image = cv2.resize(image, (self.image_size, self.image_size))
        
        # 确保是uint8格式
        if image.dtype != np.uint8:
            image = image_tools.convert_to_uint8(image)
        
        return image
    
    def create_openpi_input(self, obs: PoolObservation, task: PoolTaskDefinition) -> Dict[str, Any]:
        """创建OpenPI模型输入"""
        # 按照OpenPI期望的格式创建输入
        return {
            "observation/exterior_image_1_left": obs.overhead_image,
            "observation/wrist_image_left": obs.wrist_image,
            "observation/state": np.concatenate([
                obs.ee_position,            # [3] 位置
                obs.ee_orientation[:3],     # [3] 方向(转轴角)
                obs.joint_positions[:1]     # [1] 夹爪位置
            ]),                             # 总共 [7] 维状态
            "prompt": task.get_description()
        }


class ActionProcessor:
    """动作处理器"""
    
    def __init__(self):
        self.action_scale = 0.1  # 动作缩放因子
        self.joint_limits = np.array([
            [-2.8973, 2.8973],  # joint 1
            [-1.7628, 1.7628],  # joint 2
            [-2.8973, 2.8973],  # joint 3
            [-3.0718, -0.0698], # joint 4
            [-2.8973, 2.8973],  # joint 5
            [-0.0175, 3.7525],  # joint 6
            [-2.8973, 2.8973]   # joint 7
        ])
    
    def process_openpi_action(self, openpi_action: np.ndarray, current_state: PoolObservation) -> PoolAction:
        """处理OpenPI输出的动作"""
        # OpenPI输出通常是 [7] 维：[x, y, z, rx, ry, rz, gripper]
        if len(openpi_action) >= 7:
            # 解析动作
            delta_pos = openpi_action[:3] * self.action_scale
            delta_orn = openpi_action[3:6] * self.action_scale
            gripper = openpi_action[6]
            
            # 计算目标关节位置（简化版本）
            target_joints = self._ik_solve(
                current_state.ee_position + delta_pos,
                current_state.ee_orientation,
                current_state.joint_positions
            )
            
            return PoolAction(
                joint_targets=target_joints,
                gripper_action=gripper
            )
        else:
            # 直接关节控制
            return PoolAction(
                joint_targets=current_state.joint_positions + openpi_action * self.action_scale,
                gripper_action=0.0
            )
    
    def _ik_solve(self, target_pos: np.ndarray, target_orn: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        """简化的逆运动学求解"""
        # 这里应该实现真正的IK，暂时返回当前关节位置的小幅度变化
        delta = np.random.uniform(-0.1, 0.1, 7)  # 简化版本
        new_joints = current_joints + delta
        
        # 限制关节范围
        for i in range(len(new_joints)):
            if i < len(self.joint_limits):
                new_joints[i] = np.clip(new_joints[i], 
                                       self.joint_limits[i][0], 
                                       self.joint_limits[i][1])
        
        return new_joints


class PoolPolicyInterface:
    """台球策略接口主类"""
    
    def __init__(
        self,
        env: PoolRobotEnvironment,
        policy_host: str = "localhost",
        policy_port: int = 8000,
        image_size: int = 224
    ):
        self.env = env
        self.policy_host = policy_host
        self.policy_port = policy_port
        
        # 初始化处理器
        self.obs_processor = ObservationProcessor(image_size)
        self.action_processor = ActionProcessor()
        
        # 初始化策略客户端
        self.policy_client = websocket_client_policy.WebsocketClientPolicy(
            policy_host, policy_port
        )
        
        # 状态管理
        self.current_task = None
        self.episode_step = 0
        self.total_reward = 0.0
        
        # 历史记录
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
    
    def set_task(self, task: PoolTaskDefinition):
        """设置当前任务"""
        self.current_task = task
        print(f"设置任务: {task.get_description()}")
    
    def reset_episode(self):
        """重置回合"""
        self.episode_step = 0
        self.total_reward = 0.0
        self.observation_history.clear()
        self.action_history.clear()
        self.reward_history.clear()
        
        # 重置环境
        raw_obs = self.env.reset()
        return self.obs_processor.process_observation(raw_obs)
    
    def step(self, max_steps: int = 1000) -> Dict[str, Any]:
        """执行一步或完整回合"""
        if self.current_task is None:
            raise ValueError("必须先设置任务")
        
        obs = self.reset_episode()
        done = False
        info = {"success": False, "reason": "unknown"}
        
        while not done and self.episode_step < max_steps:
            # 获取策略动作
            action = self._get_policy_action(obs)
            
            # 执行动作
            raw_obs, reward, done, step_info = self.env.step(action.joint_targets)
            
            # 处理观测
            obs = self.obs_processor.process_observation(raw_obs)
            
            # 记录历史
            self.observation_history.append(obs)
            self.action_history.append(action)
            self.reward_history.append(reward)
            self.total_reward += reward
            self.episode_step += 1
            
            # 检查任务完成
            task_success = self._check_task_completion(obs, step_info)
            if task_success:
                done = True
                info = {"success": True, "reason": "task_completed"}
            
            # 打印进度
            if self.episode_step % 50 == 0:
                print(f"步骤 {self.episode_step}: 奖励={reward:.3f}, 总奖励={self.total_reward:.3f}")
        
        # 生成最终结果
        return {
            "total_reward": self.total_reward,
            "episode_length": self.episode_step,
            "success": info["success"],
            "task_description": self.current_task.get_description(),
            "performance_metrics": self._calculate_metrics(obs)
        }
    
    def _get_policy_action(self, obs: PoolObservation) -> PoolAction:
        """获取策略动作"""
        try:
            # 创建OpenPI输入
            openpi_input = self.obs_processor.create_openpi_input(obs, self.current_task)
            
            # 查询策略
            policy_output = self.policy_client.infer(openpi_input)
            
            # 处理输出
            if "actions" in policy_output:
                actions = policy_output["actions"]
                if len(actions) > 0:
                    # 取第一个动作
                    action = actions[0] if isinstance(actions[0], np.ndarray) else np.array(actions[0])
                    return self.action_processor.process_openpi_action(action, obs)
            
            # 后备策略：随机小幅度动作
            print("Warning: 策略输出无效，使用随机动作")
            random_action = np.random.uniform(-0.1, 0.1, 7)
            return self.action_processor.process_openpi_action(random_action, obs)
            
        except Exception as e:
            print(f"策略查询失败: {e}")
            # 使用安全的默认动作
            return PoolAction(
                joint_targets=obs.joint_positions,
                gripper_action=0.0
            )
    
    def _check_task_completion(self, obs: PoolObservation, step_info: Dict[str, Any]) -> bool:
        """检查任务是否完成"""
        if not self.current_task:
            return False
        
        success_criteria = self.current_task.get_success_criteria()
        
        if self.current_task.task_type == "pot_ball":
            balls_potted = step_info.get("balls_potted", 0)
            required_balls = success_criteria.get("balls_potted", 1)
            return balls_potted >= required_balls
        
        elif self.current_task.task_type == "break_shot":
            # 检查是否有足够的球被移动
            moved_balls = self._count_moved_balls(obs)
            return moved_balls >= success_criteria.get("min_balls_moved", 4)
        
        return False
    
    def _count_moved_balls(self, obs: PoolObservation) -> int:
        """统计移动的球数量"""
        moved_count = 0
        for ball_id, velocity in obs.ball_velocities.items():
            if np.linalg.norm(velocity) > 0.01:  # 速度阈值
                moved_count += 1
        return moved_count
    
    def _calculate_metrics(self, final_obs: PoolObservation) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {
            "episode_length": self.episode_step,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / max(1, self.episode_step),
        }
        
        if self.current_task:
            if self.current_task.task_type == "pot_ball":
                # 计算精度相关指标
                cue_ball_pos = final_obs.ball_positions.get("cue", np.zeros(3))
                table_center = np.array([0, 0, 0])  # 假设台球桌中心
                distance_to_center = np.linalg.norm(cue_ball_pos[:2] - table_center[:2])
                metrics["cue_ball_position_score"] = max(0, 1 - distance_to_center / 2.0)
        
        return metrics
    
    def run_evaluation(self, tasks: List[PoolTaskDefinition], num_trials: int = 5) -> Dict[str, Any]:
        """运行完整评估"""
        results = {
            "tasks": [],
            "overall_success_rate": 0.0,
            "overall_average_reward": 0.0
        }
        
        total_success = 0
        total_reward = 0.0
        total_trials = 0
        
        for task in tasks:
            task_results = {
                "task_description": task.get_description(),
                "trials": [],
                "success_rate": 0.0,
                "average_reward": 0.0
            }
            
            task_success = 0
            task_reward = 0.0
            
            for trial in range(num_trials):
                print(f"\n=== 任务: {task.get_description()} - 试验 {trial+1}/{num_trials} ===")
                
                self.set_task(task)
                trial_result = self.step()
                
                task_results["trials"].append(trial_result)
                
                if trial_result["success"]:
                    task_success += 1
                    total_success += 1
                
                task_reward += trial_result["total_reward"]
                total_reward += trial_result["total_reward"]
                total_trials += 1
                
                print(f"试验结果: 成功={trial_result['success']}, 奖励={trial_result['total_reward']:.2f}")
            
            task_results["success_rate"] = task_success / num_trials
            task_results["average_reward"] = task_reward / num_trials
            results["tasks"].append(task_results)
            
            print(f"任务完成: 成功率={task_results['success_rate']:.2%}, 平均奖励={task_results['average_reward']:.2f}")
        
        results["overall_success_rate"] = total_success / total_trials if total_trials > 0 else 0.0
        results["overall_average_reward"] = total_reward / total_trials if total_trials > 0 else 0.0
        
        return results


if __name__ == "__main__":
    # 测试台球OpenPI接口
    print("测试台球OpenPI接口...")
    
    try:
        # 创建环境
        env = PoolRobotEnvironment(gui=False)  # 无GUI测试
        print("✅ 环境创建成功")
        
        # 创建策略接口
        policy_interface = PoolPolicyInterface(env)
        print("✅ 策略接口创建成功")
        
        # 创建测试任务
        test_tasks = [
            PoolTaskDefinition("pot_ball", target_ball="1", target_pocket="corner"),
            PoolTaskDefinition("break_shot"),
        ]
        
        # 运行简单测试
        for task in test_tasks:
            print(f"\n测试任务: {task.get_description()}")
            policy_interface.set_task(task)
            
            # 重置并获取观测
            obs = policy_interface.reset_episode()
            print(f"✅ 观测获取成功: {type(obs)}")
            
            # 模拟几步
            for step in range(3):
                action = PoolAction(
                    joint_targets=obs.joint_positions + np.random.uniform(-0.01, 0.01, 7),
                    gripper_action=0.0
                )
                raw_obs, reward, done, info = env.step(action.joint_targets)
                obs = policy_interface.obs_processor.process_observation(raw_obs)
                print(f"步骤 {step+1}: 奖励={reward:.3f}")
        
        env.close()
        print("✅ 台球OpenPI接口测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 