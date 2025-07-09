#!/usr/bin/env python3
"""
真正的Pooltool台球机器人演示

基于pooltool专业台球物理引擎的完整集成演示：
- Pooltool: 专业台球物理、轨迹计算、碰撞检测
- Franka Panda: 7-DOF机械臂精确控制
- 协同仿真: 机械臂定位 + 专业击球物理

版本: 2.0.0 - 真正的pooltool集成
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
from typing import Dict, List, Optional, Tuple

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 导入我们的真正pooltool集成
from physics_bridge import TruePooltoolBridge, CueAction, BallState

class TruePooltoolRobotDemo:
    """
    真正的Pooltool台球机器人演示
    
    特点:
    - 使用pooltool专业台球物理引擎
    - Franka Panda 7-DOF机械臂精确控制
    - 真实台球规则和物理效应
    - 专业轨迹预测和碰撞检测
    """
    
    def __init__(self, use_gui=True, record_video=False):
        """
        初始化真正的pooltool机器人演示
        
        Args:
            use_gui: 是否使用GUI显示
            record_video: 是否录制视频
        """
        self.use_gui = use_gui
        self.record_video = record_video
        self.video_writer = None
        
        # 初始化PyBullet (仅用于机械臂渲染)
        if use_gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # 设置PyBullet环境
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
        
        # 创建简单环境 (地面等)
        self._create_environment()
        
        # 初始化真正的pooltool集成系统 🎯
        print("🎯 初始化真正的Pooltool台球物理引擎...")
        self.pooltool_bridge = TruePooltoolBridge(
            physics_client_id=self.physics_client,
            table_type="POCKET",  # 专业美式台球桌
            robot_position=(-1.2, 0, 0.83),
            enable_3d_viz=True
        )
        
        # 演示参数
        self.demo_shots = [
            CueAction(velocity=4.0, phi=0.0, theta=0.0, offset_a=0.0, offset_b=0.0),      # 直击
            CueAction(velocity=5.5, phi=np.pi/6, theta=0.0, offset_a=0.01, offset_b=0.0), # 斜击
            CueAction(velocity=3.2, phi=-np.pi/8, theta=0.0, offset_a=0.0, offset_b=0.01), # 旋转球
            CueAction(velocity=6.0, phi=np.pi/4, theta=0.05, offset_a=-0.01, offset_b=0.0), # 跳球
        ]
        
        print("✅ 真正的Pooltool台球机器人演示初始化完成!")
    
    def _create_environment(self):
        """创建简单的PyBullet环境 (仅用于机械臂背景)"""
        # 地面
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        
        # 设置材质
        p.changeDynamics(
            plane_id, -1,
            lateralFriction=0.1,
            restitution=0.1,
            physicsClientId=self.physics_client
        )
        
        print("✅ PyBullet环境创建完成 (仅用于机械臂渲染)")
    
    def setup_camera_view(self, preset: str = "overview"):
        """设置摄像头视角"""
        camera_configs = {
            "overview": {
                "distance": 3.5,
                "yaw": 45,
                "pitch": -30,
                "target": [0, 0, 0.8]
            },
            "table_side": {
                "distance": 2.8,
                "yaw": 90,
                "pitch": -20,
                "target": [0, 0, 0.8]
            },
            "robot_view": {
                "distance": 2.0,
                "yaw": -45,
                "pitch": -25,
                "target": [-1.2, 0, 0.8]
            },
            "action_close": {
                "distance": 1.8,
                "yaw": 30,
                "pitch": -15,
                "target": [-0.5, 0, 0.8]
            }
        }
        
        config = camera_configs.get(preset, camera_configs["overview"])
        
        p.resetDebugVisualizerCamera(
            cameraDistance=config["distance"],
            cameraYaw=config["yaw"],
            cameraPitch=config["pitch"],
            cameraTargetPosition=config["target"],
            physicsClientId=self.physics_client
        )
        
        print(f"📷 摄像头设置为 '{preset}' 视角")
    
    def setup_video_recording(self, video_filename: str):
        """设置视频录制"""
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                video_filename, fourcc, 30.0, (1920, 1080)
            )
            print(f"🎥 视频录制设置: {video_filename}")
    
    def demonstrate_pooltool_physics(self):
        """演示pooltool专业台球物理"""
        print("\n🎱 ===== 真正的Pooltool台球物理演示 =====")
        
        # 显示初始状态
        print("\n📊 初始台球桌状态:")
        initial_states = self.pooltool_bridge.get_ball_states()
        for ball_id, state in initial_states.items():
            print(f"  {ball_id}球: 位置=({state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f})")
        
        print(f"\n总球数: {len(initial_states)} (包括主球)")
        
        # 执行多种击球演示
        for i, shot in enumerate(self.demo_shots, 1):
            print(f"\n🏌️ 第{i}次击球 - pooltool专业物理仿真")
            print(f"  参数: 速度={shot.velocity:.1f}m/s, 角度={np.degrees(shot.phi):.1f}°")
            print(f"  偏移: a={shot.offset_a:.3f}, b={shot.offset_b:.3f}")
            
            # 机械臂准备动作
            self._perform_robot_setup_sequence(i)
            
            # 使用pooltool执行击球 🎯
            result = self.pooltool_bridge.execute_shot(
                velocity=shot.velocity,
                angle_deg=np.degrees(shot.phi),
                offset_x=shot.offset_a,
                offset_y=shot.offset_b
            )
            
            # 分析结果
            self._analyze_shot_result(result, i)
            
            # 等待观察
            print(f"  ⏱️ 等待{3}秒观察结果...")
            for _ in range(3 * 240):  # 3秒 @ 240Hz
                p.stepSimulation(physicsClientId=self.physics_client)
                if self.use_gui:
                    time.sleep(1/240)
            
            # 如果不是最后一击，重置台球桌
            if i < len(self.demo_shots):
                print(f"  🔄 重置台球桌准备下一击...")
                self.pooltool_bridge.reset_table()
                time.sleep(1)
    
    def _perform_robot_setup_sequence(self, shot_number: int):
        """执行机械臂准备动作序列"""
        print(f"  🤖 Franka机械臂准备第{shot_number}次击球...")
        
        # 不同击球的不同机械臂姿态
        robot_poses = [
            [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.8],      # 标准击球姿态
            [0.3, -0.5, 0.2, -1.8, 0.1, 1.5, -0.2],     # 右侧击球姿态  
            [-0.2, -0.2, -0.3, -2.0, -0.1, 1.8, 0.5],   # 左侧击球姿态
            [0.1, -0.4, 0.1, -1.9, 0.0, 1.6, 0.0],      # 精确击球姿态
        ]
        
        pose_idx = (shot_number - 1) % len(robot_poses)
        target_pose = robot_poses[pose_idx]
        
        # 执行机械臂移动
        self.pooltool_bridge.move_robot(target_pose, duration=1.5)
        print(f"  ✅ 机械臂已就位，准备击球")
    
    def _analyze_shot_result(self, result: Dict, shot_number: int):
        """分析击球结果"""
        print(f"  📈 第{shot_number}次击球结果分析 (基于pooltool专业物理):")
        
        # 碰撞分析
        collisions = result['collisions']
        if collisions:
            print(f"    💥 检测到{len(collisions)}次球球碰撞:")
            for ball1, ball2 in collisions:
                print(f"      - {ball1}球 与 {ball2}球 碰撞")
        else:
            print("    💥 未检测到球球碰撞")
        
        # 进袋分析
        pocketed = result['pocketed_balls']
        if pocketed:
            print(f"    🕳️ 进袋球: {', '.join(pocketed)}")
        else:
            print("    🕳️ 无球进袋")
        
        # 球状态分析
        ball_states = result['ball_states']
        active_balls = [bid for bid, state in ball_states.items() if state.active]
        print(f"    🎱 桌面剩余球数: {len(active_balls)}")
        
        # 主球最终状态
        if 'cue' in ball_states:
            cue_state = ball_states['cue']
            final_speed = np.linalg.norm(cue_state.velocity)
            print(f"    ⚪ 主球最终速度: {final_speed:.2f}m/s")
            print(f"    ⚪ 主球最终位置: ({cue_state.position[0]:.2f}, {cue_state.position[1]:.2f})")
        
        print(f"  ✅ 击球分析完成")
    
    def demonstrate_3d_visualization(self):
        """演示pooltool 3D可视化"""
        print("\n🎨 启动Pooltool专业3D可视化界面...")
        try:
            # 使用pooltool的原生3D可视化
            self.pooltool_bridge.show_3d_visualization()
        except Exception as e:
            print(f"⚠️ 3D可视化启动失败: {e}")
            print("这可能是由于WSL/无头环境的显示限制")
            print("仿真数据已生成，可以导出查看")
    
    def export_simulation_data(self):
        """导出仿真数据"""
        timestamp = int(time.time())
        data_dir = current_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        # 导出pooltool系统状态
        save_path = data_dir / f"true_pooltool_simulation_{timestamp}.json"
        self.pooltool_bridge.save_simulation(str(save_path))
        
        # 导出击球历史
        shot_history = self.pooltool_bridge.get_shot_history()
        history_path = data_dir / f"shot_history_{timestamp}.json"
        
        import json
        try:
            # 简化历史数据用于JSON序列化
            simplified_history = []
            for i, shot in enumerate(shot_history):
                simplified_history.append({
                    'shot_number': i + 1,
                    'velocity': shot['shot_params'].velocity,
                    'angle_deg': np.degrees(shot['shot_params'].phi),
                    'offset_a': shot['shot_params'].offset_a,
                    'offset_b': shot['shot_params'].offset_b,
                    'timestamp': shot['timestamp']
                })
            
            with open(history_path, 'w') as f:
                json.dump(simplified_history, f, indent=2)
            
            print(f"📊 击球历史已导出: {history_path}")
            
        except Exception as e:
            print(f"⚠️ 击球历史导出失败: {e}")
        
        return save_path
    
    def run_complete_demo(self, duration: float = 30.0):
        """运行完整的真正pooltool演示"""
        print("🚀 开始真正的Pooltool台球机器人完整演示...")
        print("=" * 60)
        
        # 设置摄像头
        self.setup_camera_view("overview")
        
        # 视频录制
        if self.record_video:
            timestamp = int(time.time())
            video_path = f"videos/true_pooltool_demo_{timestamp}.mp4"
            self.setup_video_recording(video_path)
        
        try:
            # 主要演示序列
            self.demonstrate_pooltool_physics()
            
            # 3D可视化演示
            print("\n" + "=" * 40)
            self.demonstrate_3d_visualization()
            
            # 数据导出
            print("\n" + "=" * 40)
            exported_path = self.export_simulation_data()
            
            print("\n🎉 真正的Pooltool台球机器人演示完成!")
            print("=" * 60)
            print(f"💡 主要成果:")
            print(f"  ✅ 使用pooltool专业台球物理引擎")
            print(f"  ✅ Franka Panda 7-DOF机械臂协同控制")
            print(f"  ✅ {len(self.demo_shots)}种不同击球技术演示")
            print(f"  ✅ 专业级碰撞检测和轨迹计算")
            print(f"  ✅ 完整仿真数据导出: {exported_path.name}")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        except Exception as e:
            print(f"\n❌ 演示过程出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.video_writer:
                self.video_writer.release()
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.video_writer:
                self.video_writer.release()
            p.disconnect(self.physics_client)
            print("✅ 资源清理完成")
        except:
            pass

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='真正的Pooltool台球机器人演示')
    parser.add_argument('--no-gui', action='store_true', help='无GUI模式运行')
    parser.add_argument('--duration', type=float, default=30.0, help='演示持续时间（秒）')
    parser.add_argument('--record', action='store_true', help='录制演示视频')
    parser.add_argument('--physics-only', action='store_true', help='仅演示物理引擎（跳过3D可视化）')
    
    args = parser.parse_args()
    
    print("🎯 真正的Pooltool台球机器人演示系统")
    print("🔬 专业台球物理 + Franka机械臂协同仿真")
    print("=" * 60)
    
    # 创建并运行演示
    demo = TruePooltoolRobotDemo(
        use_gui=not args.no_gui,
        record_video=args.record
    )
    
    try:
        if args.physics_only:
            # 仅演示物理引擎
            demo.demonstrate_pooltool_physics()
            demo.export_simulation_data()
        else:
            # 完整演示
            demo.run_complete_demo(duration=args.duration)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main() 