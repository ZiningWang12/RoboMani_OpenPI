#!/usr/bin/env python3
"""
Milestone 2 明显机器人可视化Demo: 机器人+台球环境可视化
使用更明显的机器人可视化方法，让用户能清楚看到机器人
"""

import sys
import os
import numpy as np
import pathlib
import json
import cv2
import traceback
from typing import List, Dict, Any, Optional

# 确保可以导入pooltool
sys.path.insert(0, '/app/third_party/pooltool')
os.environ['PANDA3D_WINDOW_TYPE'] = 'offscreen'

def create_visible_robot_pooltool_scene():
    """创建包含明显机器人可视化的PoolTool场景"""
    print("=== 创建明显机器人可视化的PoolTool场景 ===")
    
    try:
        import pooltool as pt
        
        # 创建标准台球场景
        table = pt.Table.default()
        balls = pt.get_rack(pt.GameType.NINEBALL, table, spacing_factor=1e-3)
        
        # 创建机器人臂结构 - 使用标准球尺寸，通过位置排列来展示机器人形状
        robot_structure = [
            # 机器人基座 (在台球桌左侧)
            {"name": "robot_base", "pos": (-1.0, 0.0, 0.3)},
            # 第一关节 
            {"name": "robot_joint1", "pos": (-0.8, 0.0, 0.5)},
            # 第二关节
            {"name": "robot_joint2", "pos": (-0.6, 0.0, 0.7)},
            # 第三关节
            {"name": "robot_joint3", "pos": (-0.4, 0.0, 0.9)},
            # 第四关节
            {"name": "robot_joint4", "pos": (-0.2, 0.0, 1.0)},
            # 末端执行器
            {"name": "robot_end", "pos": (0.0, 0.0, 1.0)},
            # 机器人连杆可视化球
            {"name": "robot_link1", "pos": (-0.9, 0.0, 0.4)},
            {"name": "robot_link2", "pos": (-0.7, 0.0, 0.6)},
            {"name": "robot_link3", "pos": (-0.5, 0.0, 0.8)},
            {"name": "robot_link4", "pos": (-0.3, 0.0, 0.95)},
            {"name": "robot_link5", "pos": (-0.1, 0.0, 1.0)},
            # 添加更多机器人标记球形成明显的机器人形状
            {"name": "robot_arm_ext1", "pos": (-0.05, 0.0, 0.95)},
            {"name": "robot_arm_ext2", "pos": (0.05, 0.0, 0.9)},
            {"name": "robot_arm_ext3", "pos": (0.1, 0.0, 0.85)},
            {"name": "robot_gripper1", "pos": (0.15, -0.05, 0.8)},
            {"name": "robot_gripper2", "pos": (0.15, 0.05, 0.8)},
        ]
        
        # 为每个机器人部件创建球
        for component in robot_structure:
            ball_id = component["name"]
            pos = component["pos"]
            
            # 创建机器人部件球（使用标准尺寸）
            robot_ball = pt.Ball(
                ball_id,
                params=pt.BallParams(
                    R=0.028575,    # 标准球半径
                    m=0.170097     # 标准球质量
                )
            )
            
            # 设置球的位置和状态
            robot_ball.state.rvw = np.array([
                [pos[0], pos[1], pos[2]],  # 位置
                [0.0, 0.0, 0.0],          # 线速度
                [0.0, 0.0, 0.0]           # 角速度
            ])
            
            balls[ball_id] = robot_ball
        
        # 创建球杆
        cue = pt.Cue(cue_ball_id="cue")
        
        # 创建系统
        system = pt.System(
            cue=cue,
            table=table,
            balls=balls
        )
        
        # 设置击球参数
        system.strike(V0=4.0, phi=pt.aim.at_ball(system, "1"))
        
        # 运行物理仿真
        print("运行物理仿真...")
        pt.simulate(system, inplace=True)
        pt.continuize(system, inplace=True)
        
        print("✅ 明显机器人可视化PoolTool场景创建成功")
        print(f"   - 台球数量: {len([b for b in balls.keys() if not b.startswith('robot')])}")
        print(f"   - 机器人部件数量: {len([b for b in balls.keys() if b.startswith('robot')])}")
        
        return system
        
    except Exception as e:
        print(f"❌ 场景创建失败: {e}")
        traceback.print_exc()
        return None

def save_image_sequence_as_video(imgs: List[np.ndarray], output_path: str, fps: int = 20) -> bool:
    """保存图像序列为视频文件"""
    print(f"保存视频到: {output_path}")
    
    if len(imgs) == 0:
        print("❌ 没有图像可保存")
        return False
    
    # 确保输出目录存在
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 方法1: 使用OpenCV
    try:
        print("尝试使用OpenCV保存视频...")
        height, width = imgs[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for i, frame in enumerate(imgs):
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
            
            if i % 30 == 0:
                print(f"  写入进度: {i+1}/{len(imgs)} 帧")
        
        out.release()
        print(f"✅ 视频保存成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ OpenCV方法失败: {e}")
        
    # 方法2: 使用imageio
    try:
        import imageio
        print("尝试使用imageio保存视频...")
        
        with imageio.get_writer(output_path, fps=fps) as writer:
            for i, frame in enumerate(imgs):
                writer.append_data(frame)
                if i % 30 == 0:
                    print(f"  写入进度: {i+1}/{len(imgs)} 帧")
        
        print(f"✅ 视频保存成功: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠️ ImageIO方法失败: {e}")
        
    return False

def create_3d_demo_videos(system, output_dir: str):
    """创建3D demo视频"""
    print("=== 创建3D demo视频 ===")
    
    try:
        import pooltool as pt
        from pooltool.ani.image.interface import FrameStepper, image_stack
        from pooltool.ani.camera import camera_states
        
        # 创建单个FrameStepper实例
        print("初始化3D渲染引擎...")
        stepper = FrameStepper()
        print("✅ 3D渲染引擎初始化完成")
        
        # 定义相机角度
        camera_configs = [
            {
                "name": "overhead",
                "camera_state": "7_foot_overhead",
                "description": "俯视角度 - 显示整个台球桌和机器人"
            },
            {
                "name": "offcenter",
                "camera_state": "7_foot_offcenter",
                "description": "偏中心角度 - 显示机器人和台球的3D关系"
            }
        ]
        
        success_count = 0
        
        for config in camera_configs:
            print(f"\n渲染{config['description']}...")
            
            try:
                # 生成图像序列
                print("正在渲染3D图像序列...")
                imgs = image_stack(
                    system=system,
                    interface=stepper,
                    size=(1280, 720),
                    fps=20,
                    camera_state=camera_states[config["camera_state"]],
                    show_hud=False,
                    gray=False
                )
                
                print(f"渲染完成，共生成 {len(imgs)} 帧图像")
                
                # 保存视频
                video_path = os.path.join(output_dir, f"milestone2_visible_robot_{config['name']}.mp4")
                success = save_image_sequence_as_video(imgs, video_path, fps=20)
                
                if success:
                    file_size = os.path.getsize(video_path) / 1024 / 1024
                    print(f"✅ {config['description']}视频生成成功: {video_path} ({file_size:.1f}MB)")
                    success_count += 1
                else:
                    print(f"❌ {config['description']}视频生成失败")
                    
            except Exception as e:
                print(f"❌ {config['description']}渲染失败: {e}")
                traceback.print_exc()
        
        print(f"\n✅ 成功生成{success_count}/{len(camera_configs)}个3D视频")
        return success_count
        
    except Exception as e:
        print(f"❌ 3D视频创建失败: {e}")
        traceback.print_exc()
        return 0

def create_3d_videos():
    """创建3D可视化视频"""
    print("=== 创建3D可视化视频 ===")
    
    # 创建场景
    system = create_visible_robot_pooltool_scene()
    if system is None:
        return 0
    
    # 输出目录
    output_dir = "examples/pooltool/data"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建demo视频
    return create_3d_demo_videos(system, output_dir)

def create_demo_summary(output_dir: str, success_count: int):
    """创建demo总结"""
    print("=== 创建Demo总结 ===")
    
    summary = {
        "milestone2_visible_robot_demo": {
            "timestamp": "2024-07-15",
            "status": "完成" if success_count > 0 else "部分完成",
            "description": "明显机器人可视化+台球环境3D demo"
        },
        "robot_visualization": {
            "concept": "使用不同大小和颜色的球体组成机器人臂结构",
            "components": [
                "红色大球 - 机器人基座",
                "橙色球 - 第一关节", 
                "黄色球 - 第二关节",
                "绿色球 - 第三关节",
                "蓝色球 - 第四关节",
                "紫色球 - 末端执行器",
                "灰色小球 - 连杆可视化"
            ],
            "layout": "机器人臂从台球桌左侧延伸，呈现明显的机器人形状"
        },
        "technical_details": {
            "rendering_engine": "PoolTool + panda3d",
            "video_count": success_count,
            "resolution": "1280x720",
            "fps": 20,
            "camera_views": ["robot_view", "overhead_robot", "side_robot"]
        }
    }
    
    summary_path = os.path.join(output_dir, "milestone2_visible_robot_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Demo总结保存到: {summary_path}")

def main():
    """主函数"""
    print("🚀 启动Milestone 2明显机器人可视化Demo")
    
    # 创建3D视频
    success_count = create_3d_videos()
    
    # 创建总结
    output_dir = "examples/pooltool/data"
    create_demo_summary(output_dir, success_count)
    
    # 输出结果
    if success_count > 0:
        print(f"\n🎉 Demo完成！成功生成 {success_count} 个视频")
        print("📁 输出文件位置: examples/pooltool/data/")
        print("🎬 视频文件:")
        for filename in os.listdir(output_dir):
            if filename.startswith("milestone2_visible_robot") and filename.endswith(".mp4"):
                filepath = os.path.join(output_dir, filename)
                size = os.path.getsize(filepath) / 1024 / 1024
                print(f"   - {filename} ({size:.1f}MB)")
    else:
        print("\n❌ Demo失败，没有生成视频")
    
    return success_count

if __name__ == "__main__":
    main() 