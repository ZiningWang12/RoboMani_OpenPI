#!/usr/bin/env python3
"""
专门的演示视频生成脚本
生成Franka Panda机械臂和Pooltool台球系统的演示视频
"""

import time
import numpy as np
import pybullet as p
import cv2
from pathlib import Path
import json

# 获取当前脚本目录
current_dir = Path(__file__).parent

class DemoVideoCreator:
    """演示视频创建器"""
    
    def __init__(self):
        self.video_writer = None
        self.video_path = None
        self.robot_id = None
        self.table_id = None
        self.physics_client = None
        
    def init_simulation(self):
        """初始化仿真环境"""
        print("🎬 初始化演示仿真环境...")
        
        # 初始化PyBullet
        self.physics_client = p.connect(p.GUI)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        
        # 加载地面
        p.loadURDF("plane.urdf")
        
        # 加载台球桌(简化版)
        table_pos = [0, 0, 0.4]
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.0, 2.0, 0.05])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.0, 2.0, 0.05], 
                                         rgbaColor=[0.0, 0.5, 0.0, 1.0])
        self.table_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_collision,
                                        baseVisualShapeIndex=table_visual, basePosition=table_pos)
        
        # 加载Franka机械臂
        try:
            franka_urdf = current_dir / "data/pybullet-panda/data/franka/panda_arm.urdf"
            if franka_urdf.exists():
                self.robot_id = p.loadURDF(str(franka_urdf), 
                                         basePosition=[-1.2, 0, 0.83],
                                         baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
                print("✅ Franka机械臂加载成功")
            else:
                print("⚠️ 未找到Franka URDF，使用简化机械臂")
                self._create_simple_robot()
        except Exception as e:
            print(f"⚠️ Franka加载失败: {e}，使用简化机械臂")
            self._create_simple_robot()
        
        # 添加一些台球
        self._add_pool_balls()
        
        # 设置摄像头
        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,
            cameraYaw=45,
            cameraPitch=-35,
            cameraTargetPosition=[0, 0, 0.8]
        )
        
        print("✅ 仿真环境初始化完成")
    
    def _create_simple_robot(self):
        """创建简化的机械臂模型"""
        # 基座
        base_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.1, height=0.2)
        base_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.1, length=0.2,
                                        rgbaColor=[0.7, 0.7, 0.7, 1.0])
        
        # 创建多体机械臂
        self.robot_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=[-1.2, 0, 0.93]
        )
        print("✅ 简化机械臂创建完成")
    
    def _add_pool_balls(self):
        """添加台球"""
        ball_positions = [
            [0.4, 0.0, 0.5],   # 白球
            [0.8, 0.0, 0.5],   # 目标球
            [0.9, 0.1, 0.5],   # 其他球
            [0.9, -0.1, 0.5],
            [1.0, 0.0, 0.5],
        ]
        
        ball_colors = [
            [1.0, 1.0, 1.0, 1.0],  # 白色
            [1.0, 0.0, 0.0, 1.0],  # 红色
            [0.0, 0.0, 1.0, 1.0],  # 蓝色
            [1.0, 1.0, 0.0, 1.0],  # 黄色
            [0.0, 1.0, 0.0, 1.0],  # 绿色
        ]
        
        for i, (pos, color) in enumerate(zip(ball_positions, ball_colors)):
            ball_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.028)
            ball_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.028, rgbaColor=color)
            
            ball_id = p.createMultiBody(
                baseMass=0.17,  # 台球重量约170g
                baseCollisionShapeIndex=ball_collision,
                baseVisualShapeIndex=ball_visual,
                basePosition=pos
            )
            
            # 设置物理属性
            p.changeDynamics(ball_id, -1, 
                           lateralFriction=0.4,
                           rollingFriction=0.01,
                           restitution=0.9)
        
        print("✅ 台球添加完成")
    
    def init_video_recording(self):
        """初始化视频录制"""
        timestamp = int(time.time() * 1000000) % 10000000
        video_filename = f"enhanced_franka_pool_demo_{timestamp}.mp4"
        self.video_path = current_dir / "videos" / video_filename
        self.video_path.parent.mkdir(exist_ok=True)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.video_path), fourcc, 30.0, (1024, 768)
        )
        print(f"📹 视频录制初始化: {video_filename}")
    
    def capture_frame(self):
        """捕获当前帧"""
        if not self.video_writer:
            return
            
        try:
            # 获取摄像头图像
            img_arr = p.getCameraImage(
                width=1024,
                height=768,
                renderer=p.ER_BULLET_HARDWARE_OPENGL
            )
            
            # 转换图像格式
            rgba_img = img_arr[2]
            rgb_img = rgba_img[:, :, :3]
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # 写入视频
            self.video_writer.write(bgr_img)
            
        except Exception as e:
            print(f"⚠️ 帧捕获失败: {e}")
    
    def create_demo_sequence(self):
        """创建演示序列"""
        print("\n🎬 开始录制演示序列...")
        
        # 场景1: 总览介绍 (3秒)
        print("📋 场景1: 系统总览")
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0, cameraYaw=30, cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.8]
        )
        self._record_scene(duration=3.0, fps=30)
        
        # 场景2: 机械臂特写 (3秒)
        print("📋 场景2: 机械臂特写")
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5, cameraYaw=-60, cameraPitch=-20,
            cameraTargetPosition=[-1.0, 0, 1.0]
        )
        self._record_scene(duration=3.0, fps=30)
        
        # 场景3: 台球桌特写 (3秒)
        print("📋 场景3: 台球桌特写")
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0, cameraYaw=90, cameraPitch=-25,
            cameraTargetPosition=[0.6, 0, 0.5]
        )
        self._record_scene(duration=3.0, fps=30)
        
        # 场景4: 模拟击球动作 (4秒)
        print("📋 场景4: 模拟击球动作")
        self._simulate_pool_shot()
        
        # 场景5: 多角度观察结果 (5秒)
        print("📋 场景5: 结果观察")
        camera_positions = [
            {"cameraDistance": 4.0, "cameraYaw": 45, "cameraPitch": -35, "cameraTargetPosition": [0, 0, 0.8]},
            {"cameraDistance": 3.5, "cameraYaw": 90, "cameraPitch": -30, "cameraTargetPosition": [0.5, 0, 0.5]},
            {"cameraDistance": 4.5, "cameraYaw": 0, "cameraPitch": -40, "cameraTargetPosition": [0, 0, 0.8]},
        ]
        
        for i, cam_config in enumerate(camera_positions):
            print(f"   👁️ 角度 {i+1}")
            p.resetDebugVisualizerCamera(**cam_config)
            self._record_scene(duration=1.7, fps=30)
        
        print("✅ 演示序列录制完成")
    
    def _record_scene(self, duration: float, fps: int = 30):
        """录制场景"""
        total_frames = int(duration * fps)
        
        for frame in range(total_frames):
            # 运行物理仿真
            p.stepSimulation()
            
            # 捕获帧
            self.capture_frame()
            
            # 控制帧率
            time.sleep(1.0 / fps)
    
    def _simulate_pool_shot(self):
        """模拟击球动作"""
        print("🎯 模拟击球动作...")
        
        # 获取白球ID (假设是第一个球)
        cue_ball_id = None
        for i in range(p.getNumBodies()):
            body_id = p.getBodyUniqueId(i)
            pos, _ = p.getBasePositionAndOrientation(body_id)
            # 寻找白球位置附近的物体
            if abs(pos[0] - 0.4) < 0.1 and abs(pos[1]) < 0.1:
                cue_ball_id = body_id
                break
        
        if cue_ball_id is not None:
            # 给白球施加力模拟击球
            p.applyExternalForce(
                cue_ball_id, -1,
                forceObj=[20, 0, 0],  # 向前的力
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME
            )
            print("✅ 击球力已施加")
        
        # 录制击球过程
        self._record_scene(duration=4.0, fps=30)
    
    def finalize_video(self):
        """完成视频录制"""
        if self.video_writer:
            self.video_writer.release()
            
            if self.video_path and self.video_path.exists():
                video_size = self.video_path.stat().st_size
                print(f"📹 视频保存成功: {self.video_path}")
                print(f"   文件大小: {video_size / 1024 / 1024:.2f} MB")
                return self.video_path
        
        return None
    
    def close(self):
        """关闭仿真"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            print("🔌 仿真环境已关闭")

def main():
    """主函数"""
    print("🎬 Franka-Pooltool 演示视频生成器")
    print("=" * 50)
    
    creator = DemoVideoCreator()
    
    try:
        # 初始化
        creator.init_simulation()
        creator.init_video_recording()
        
        # 录制演示
        creator.create_demo_sequence()
        
        # 完成
        video_path = creator.finalize_video()
        
        print(f"\n🎉 演示视频生成完成!")
        if video_path:
            print(f"   视频路径: {video_path}")
            print(f"   视频时长: 约18秒")
            print(f"   分辨率: 1024x768@30fps")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        creator.close()

if __name__ == "__main__":
    main() 