#!/usr/bin/env python3
"""
台球机器人端到端演示

这个脚本演示了：
1. PyBullet物理仿真的台球环境
2. Pooltool的3D可视化
3. 视频录制功能

使用方法:
    python main.py --mode=demo                    # 运行演示
    python main.py --mode=record --duration=10   # 录制10秒视频
    python main.py --mode=pooltool               # Pooltool可视化
"""

import argparse
import time
import numpy as np
import pybullet as p
import pooltool
from pathlib import Path

# 导入我们的模块
from physics_bridge import PhysicsBridge

class PoolTableDemo:
    """台球桌演示类"""
    
    def __init__(self, gui: bool = True, enable_pooltool: bool = True):
        self.gui = gui
        self.enable_pooltool = enable_pooltool
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化PyBullet
        self.physics_client = p.connect(p.GUI if gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(1./240., physicsClientId=self.physics_client)
        
        # 初始化物理桥接器
        self.physics_bridge = PhysicsBridge(
            self.physics_client,
            enable_pooltool_viz=enable_pooltool
        )
        
        print("✅ 台球演示环境初始化完成")
    
    def run_physics_demo(self, duration: float = 10.0):
        """运行物理演示"""
        print(f"🎱 开始物理演示 (持续 {duration} 秒)")
        
        start_time = time.time()
        step_count = 0
        
        # 在第100步时给主球一个冲量
        cue_impulse_applied = False
        
        while time.time() - start_time < duration:
            step_count += 1
            
            # 在第100步时击打主球
            if step_count == 100 and not cue_impulse_applied:
                impulse = np.array([8.0, 2.0, 0.0])  # 斜向冲量
                self.physics_bridge.apply_ball_impulse("cue", impulse)
                print("🏌️ 击打主球！")
                cue_impulse_applied = True
            
            # 在第500步时再次击打
            if step_count == 500:
                impulse = np.array([-6.0, 4.0, 0.0])  # 反向冲量
                self.physics_bridge.apply_ball_impulse("1", impulse)
                print("🏌️ 击打目标球！")
            
            # 推进物理仿真
            self.physics_bridge.step_simulation(1./240.)
            
            # 每60步检查一次状态
            if step_count % 60 == 0:
                # 检查碰撞
                collisions = self.physics_bridge.detect_ball_collisions()
                if collisions:
                    print(f"💥 球碰撞: {collisions}")
                
                # 检查进袋
                pocketed = self.physics_bridge.check_ball_pocketed()
                if pocketed:
                    print(f"🕳️ 球进袋: {pocketed}")
                
                # 获取球状态
                ball_states = self.physics_bridge.get_ball_states()
                active_balls = [bid for bid, state in ball_states.items() 
                              if state.position[2] > 0.5]  # 还在桌面上的球
                print(f"🎱 活跃球数: {len(active_balls)}/2")
            
            # 控制仿真速度
            if self.gui:
                time.sleep(1./60.)  # 60 FPS
        
        print("✅ 物理演示完成")
    
    def run_interactive_demo(self, duration: float = 30.0):
        """运行交互式演示，包含多次击球"""
        print(f"🎱 开始交互式演示 (持续 {duration} 秒)")
        
        start_time = time.time()
        step_count = 0
        
        # 定义击球序列
        shot_sequence = [
            (100, "cue", np.array([8.0, 2.0, 0.0])),   # 第1次击球
            (300, "1", np.array([-6.0, 4.0, 0.0])),    # 第2次击球
            (500, "cue", np.array([5.0, -3.0, 0.0])),  # 第3次击球
            (700, "1", np.array([-4.0, -5.0, 0.0])),   # 第4次击球
        ]
        
        shot_index = 0
        
        while time.time() - start_time < duration:
            step_count += 1
            
            # 执行击球序列
            if shot_index < len(shot_sequence):
                target_step, ball_id, impulse = shot_sequence[shot_index]
                if step_count == target_step:
                    self.physics_bridge.apply_ball_impulse(ball_id, impulse)
                    print(f"🏌️ 第{shot_index+1}次击球: {ball_id} 球!")
                    shot_index += 1
            
            # 推进物理仿真
            self.physics_bridge.step_simulation(1./240.)
            
            # 每120步检查一次状态
            if step_count % 120 == 0:
                # 检查碰撞
                collisions = self.physics_bridge.detect_ball_collisions()
                if collisions:
                    print(f"💥 球碰撞: {collisions}")
                
                # 检查进袋
                pocketed = self.physics_bridge.check_ball_pocketed()
                if pocketed:
                    print(f"🕳️ 球进袋: {pocketed}")
                
                # 获取球状态
                ball_states = self.physics_bridge.get_ball_states()
                for ball_id, state in ball_states.items():
                    pos = state.position
                    vel_magnitude = np.linalg.norm(state.velocity)
                    print(f"🎱 {ball_id}球: 位置=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), 速度={vel_magnitude:.2f}")
            
            # 控制仿真速度
            if self.gui:
                time.sleep(1./60.)  # 60 FPS
        
        print("✅ 交互式演示完成")
    
    def record_video(self, duration: float = 10.0, filename: str = None, interactive: bool = True):
        """录制演示视频"""
        if not self.gui:
            print("❌ 视频录制需要GUI模式")
            return
        
        if filename is None:
            timestamp = int(time.time())
            mode = "interactive" if interactive else "simple"
            filename = f"pool_demo_{mode}_{timestamp}.mp4"
        
        video_path = self.data_dir / filename
        
        print(f"🎥 开始录制视频: {video_path}")
        
        # 启动视频录制
        p.startStateLogging(
            p.STATE_LOGGING_VIDEO_MP4,
            str(video_path),
            physicsClientId=self.physics_client
        )
        
        try:
            # 运行演示
            if interactive:
                self.run_interactive_demo(duration)
            else:
                self.run_physics_demo(duration)
        finally:
            # 停止录制
            p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4, physicsClientId=self.physics_client)
            print(f"✅ 视频已保存: {video_path}")
    
    def create_pooltool_visualization(self):
        """创建pooltool 3D可视化"""
        if not self.enable_pooltool:
            print("❌ Pooltool可视化未启用")
            return
        
        print("🎨 创建Pooltool 3D可视化...")
        
        try:
            # 获取当前球状态
            ball_states = self.physics_bridge.get_ball_states()
            
            # 创建可视化系统
            viz_system = self.physics_bridge.create_pooltool_visualization()
            
            if viz_system:
                print("✅ Pooltool可视化系统创建成功")
                
                # 使用pooltool.show()显示3D界面
                try:
                    pooltool.show(viz_system)
                except Exception as e:
                    print(f"⚠️ Pooltool显示失败: {e}")
                    print("这可能是由于WSL环境的显示限制")
            else:
                print("❌ 无法创建Pooltool可视化系统")
                
        except Exception as e:
            print(f"❌ Pooltool可视化失败: {e}")
    
    def run_pooltool_simulation_demo(self):
        """运行pooltool原生仿真演示"""
        if not self.enable_pooltool:
            print("❌ Pooltool未启用")
            return
        
        print("🎨 运行Pooltool原生仿真演示...")
        
        try:
            # 创建一个标准台球场景
            system = pooltool.System.example()
            
            # 设置击球参数
            cue = system.cue
            cue.set_state(
                V0=5.0,  # 击球速度
                phi=0.0,  # 水平角度
                theta=0.0,  # 垂直角度
                a=0.0,    # 击球点水平偏移
                b=0.0     # 击球点垂直偏移
            )
            
            print("✅ 设置击球参数完成")
            
            # 运行仿真
            print("🎬 开始pooltool仿真...")
            
            # 使用pooltool的simulate函数
            simulated_system = pooltool.simulate(
                system, 
                inplace=False,
                continuous=True,  # 连续轨迹
                dt=0.01           # 时间步长
            )
            
            print("✅ Pooltool仿真完成")
            
            # 显示结果
            if simulated_system:
                try:
                    print("🎨 显示仿真结果...")
                    pooltool.show(simulated_system)
                except Exception as e:
                    print(f"⚠️ 显示失败: {e}")
                    print("尝试导出数据...")
                    
                    # 导出仿真数据
                    timestamp = int(time.time())
                    save_path = self.data_dir / f"pooltool_simulation_{timestamp}.json"
                    simulated_system.save(str(save_path))
                    print(f"✅ 仿真数据已保存: {save_path}")
            else:
                print("❌ 仿真失败")
                
        except Exception as e:
            print(f"❌ Pooltool仿真演示失败: {e}")
            import traceback
            traceback.print_exc()
    
    def export_pooltool_data(self):
        """导出pooltool数据用于离线可视化"""
        if not self.enable_pooltool:
            print("❌ Pooltool未启用")
            return
        
        print("💾 导出Pooltool数据...")
        
        try:
            # 获取可视化系统
            viz_system = self.physics_bridge.create_pooltool_visualization()
            
            if viz_system:
                # 保存系统状态
                timestamp = int(time.time())
                save_path = self.data_dir / f"pooltool_system_{timestamp}.json"
                
                # 使用pooltool的保存功能
                viz_system.save(str(save_path))
                print(f"✅ Pooltool数据已保存: {save_path}")
                
                return save_path
            else:
                print("❌ 无法创建Pooltool系统用于导出")
                return None
                
        except Exception as e:
            print(f"❌ 数据导出失败: {e}")
            return None
    
    def cleanup(self):
        """清理资源"""
        try:
            p.disconnect(self.physics_client)
            print("✅ 资源清理完成")
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="台球机器人演示")
    parser.add_argument("--mode", type=str, default="demo", 
                       choices=["demo", "interactive", "record", "pooltool", "simulate"],
                       help="演示模式")
    parser.add_argument("--duration", type=float, default=10.0,
                       help="演示持续时间（秒）")
    parser.add_argument("--no-gui", action="store_true",
                       help="禁用GUI")
    parser.add_argument("--no-pooltool", action="store_true",
                       help="禁用Pooltool")
    parser.add_argument("--video-file", type=str, default=None,
                       help="视频文件名")
    parser.add_argument("--simple", action="store_true",
                       help="使用简单模式（用于录制）")
    
    args = parser.parse_args()
    
    # 创建演示
    demo = PoolTableDemo(
        gui=not args.no_gui,
        enable_pooltool=not args.no_pooltool
    )
    
    try:
        if args.mode == "demo":
            print("🎱 运行基础物理演示")
            demo.run_physics_demo(args.duration)
            
        elif args.mode == "interactive":
            print("🎮 运行交互式演示")
            demo.run_interactive_demo(args.duration)
            
        elif args.mode == "record":
            print("🎥 录制演示视频")
            demo.record_video(args.duration, args.video_file, not args.simple)
            
        elif args.mode == "pooltool":
            print("🎨 创建Pooltool 3D可视化")
            demo.create_pooltool_visualization()
            # 同时导出数据
            demo.export_pooltool_data()
            
        elif args.mode == "simulate":
            print("🎬 运行Pooltool原生仿真")
            demo.run_pooltool_simulation_demo()
        
        print("\n🎉 演示完成！")
        
        # 提供一些下一步建议
        print("\n📋 可用的命令:")
        print("  python main.py --mode=demo                       # 基础物理演示")
        print("  python main.py --mode=interactive --duration=20  # 交互式演示")
        print("  python main.py --mode=record --duration=15       # 录制交互视频")
        print("  python main.py --mode=record --simple            # 录制简单视频")
        print("  python main.py --mode=pooltool                   # Pooltool可视化")
        print("  python main.py --mode=simulate                   # Pooltool仿真")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main() 