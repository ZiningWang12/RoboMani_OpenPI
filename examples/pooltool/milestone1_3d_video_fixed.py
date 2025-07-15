#!/usr/bin/env python3
"""
Milestone 1: 修正版 - 使用PoolTool原生3D渲染引擎生成台球仿真视频
解决ShowBase实例和ffmpeg问题
"""

import sys
import os
import numpy as np
import pathlib
import traceback
from typing import Optional

# 确保可以导入pooltool
sys.path.insert(0, '/app/third_party/pooltool')

# 设置panda3d为无头模式（Docker环境）
os.environ['PANDA3D_WINDOW_TYPE'] = 'offscreen'

def create_break_shot_system(scenario_name: str, V0: float = 8.0, seed: Optional[int] = None):
    """创建一个标准的台球开球系统"""
    import pooltool as pt
    
    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)
    
    print(f"正在创建 '{scenario_name}' 系统...")
    
    # 创建标准台球桌
    table = pt.Table.default()
    
    # 根据场景类型创建球的排列
    if "eightball" in scenario_name.lower():
        balls = pt.get_rack(pt.GameType.EIGHTBALL, table, spacing_factor=1e-3)
    else:
        balls = pt.get_rack(pt.GameType.NINEBALL, table, spacing_factor=1e-3)
    
    # 创建球杆
    cue = pt.Cue(cue_ball_id="cue")
    
    # 创建系统
    system = pt.System(
        cue=cue,
        table=table,
        balls=balls
    )
    
    # 设置击球参数 - 瞄准1号球
    system.strike(V0=V0, phi=pt.aim.at_ball(system, "1"))
    
    print(f"✅ '{scenario_name}' 系统创建完成")
    return system

def save_image_sequence_as_video(imgs, output_path: str, fps: int = 30):
    """将图像序列保存为视频文件"""
    try:
        # 尝试多种方法保存视频
        print(f"正在保存视频: {output_path}")
        
        # 方法1: 使用opencv-python
        try:
            import cv2
            
            if len(imgs) == 0:
                print("❌ 没有图像可以保存")
                return False
                
            # 获取图像尺寸
            height, width = imgs[0].shape[:2]
            
            # 创建VideoWriter对象
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for i, frame in enumerate(imgs):
                # 转换RGB到BGR (OpenCV使用BGR)
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
            print(f"⚠️  OpenCV方法失败: {e}")
            
        # 方法2: 使用imageio with ffmpeg
        try:
            import imageio
            
            # 安装ffmpeg后端
            import subprocess
            try:
                subprocess.run(['pip', 'install', 'imageio[ffmpeg]'], 
                             check=True, capture_output=True)
                print("✅ FFmpeg后端安装成功")
            except:
                print("⚠️  FFmpeg后端安装失败，尝试使用现有后端")
            
            with imageio.get_writer(output_path, fps=fps) as writer:
                for i, frame in enumerate(imgs):
                    writer.append_data(frame)
                    if i % 30 == 0:
                        print(f"  写入进度: {i+1}/{len(imgs)} 帧")
            
            print(f"✅ 视频保存成功: {output_path}")
            return True
            
        except Exception as e:
            print(f"⚠️  ImageIO方法失败: {e}")
            
        # 方法3: 保存为PNG序列
        png_dir = pathlib.Path(output_path).parent / f"{pathlib.Path(output_path).stem}_frames"
        png_dir.mkdir(exist_ok=True)
        
        from PIL import Image
        
        for i, frame in enumerate(imgs):
            png_path = png_dir / f"frame_{i:06d}.png"
            Image.fromarray(frame).save(png_path)
            if i % 30 == 0:
                print(f"  保存帧: {i+1}/{len(imgs)}")
        
        print(f"✅ 图像序列保存成功: {png_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 视频保存失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主函数 - 生成高质量3D台球仿真视频"""
    try:
        print("=== Milestone 1: PoolTool 3D台球仿真视频生成 (修正版) ===\n")
        
        # 导入必要的模块
        import pooltool as pt
        from pooltool.ani.image.interface import FrameStepper, image_stack
        from pooltool.ani.camera import camera_states
        
        print(f"✅ PoolTool版本: {pt.__version__}")
        
        # 创建输出目录
        output_dir = pathlib.Path("/app/examples/pooltool/data/pooltool/milestone1_3d_videos")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建单个FrameStepper实例 (避免多个ShowBase实例)
        print("正在初始化3D渲染引擎...")
        stepper = FrameStepper()
        print("✅ 3D渲染引擎初始化完成")
        
        # 定义场景
        scenarios = [
            {
                "name": "nineball_break_slow",
                "description": "九球慢速开球",
                "V0": 5.0,
                "seed": 42
            },
            {
                "name": "nineball_break_fast",
                "description": "九球快速开球",
                "V0": 10.0,
                "seed": 123
            },
            {
                "name": "eightball_break",
                "description": "八球开球",
                "V0": 8.0,
                "seed": 456
            }
        ]
        
        # 可用的相机角度
        camera_angles = ["7_foot_overhead", "7_foot_offcenter"]
        
        for scenario in scenarios:
            print(f"\n--- 处理场景: {scenario['description']} ---")
            
            # 创建系统
            system = create_break_shot_system(
                scenario_name=scenario["name"],
                V0=scenario["V0"],
                seed=scenario["seed"]
            )
            
            # 运行物理仿真
            print("正在运行物理仿真...")
            pt.simulate(system, inplace=True)
            print("✅ 物理仿真完成")
            
            # 为每个相机角度生成视频
            for angle in camera_angles:
                if angle not in camera_states:
                    print(f"⚠️  跳过不可用的相机角度: {angle}")
                    continue
                    
                print(f"正在渲染相机角度: {angle}")
                
                try:
                    # 生成图像序列
                    print("正在渲染3D图像序列...")
                    imgs = image_stack(
                        system=system,
                        interface=stepper,
                        size=(1280, 720),  # 720p分辨率
                        fps=30,
                        camera_state=camera_states[angle],
                        show_hud=False,
                        gray=False
                    )
                    
                    print(f"渲染完成，共生成 {len(imgs)} 帧图像")
                    
                    # 保存视频
                    video_path = output_dir / f"{scenario['name']}_{angle}.mp4"
                    success = save_image_sequence_as_video(
                        imgs, 
                        str(video_path),
                        fps=30
                    )
                    
                    if success:
                        print(f"✅ 视频保存成功: {scenario['name']}_{angle}")
                    else:
                        print(f"❌ 视频保存失败: {scenario['name']}_{angle}")
                        
                except Exception as e:
                    print(f"❌ 渲染失败: {e}")
                    traceback.print_exc()
        
        print(f"\n🎉 Milestone 1 - 3D视频生成完成！")
        print(f"输出目录: {output_dir}")
        
        # 列出生成的文件
        print("\n生成的文件:")
        for file_path in sorted(output_dir.glob("*")):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.1f} MB)")
            else:
                print(f"  - {file_path.name}/ (目录)")
            
    except Exception as e:
        print(f"❌ 3D视频生成过程中出错: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 