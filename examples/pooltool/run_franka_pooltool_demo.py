#!/usr/bin/env python3
"""
Franka-Pooltool协同仿真演示运行脚本

快速启动和测试完整的机械臂-台球物理集成系统

使用方法:
    python run_franka_pooltool_demo.py [--mode MODE] [--shots N] [--record]

作者: OpenPI团队
版本: 1.0.0
"""

import sys
import argparse
from pathlib import Path
import time

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from franka_pooltool_integration import FrankaPooltoolIntegration

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Franka-Pooltool协同仿真演示"
    )
    
    parser.add_argument(
        "--mode", 
        choices=["demo", "interactive", "benchmark"],
        default="demo",
        help="运行模式: demo(演示), interactive(交互), benchmark(性能测试)"
    )
    
    parser.add_argument(
        "--shots",
        type=int,
        default=4,
        help="演示击球次数 (默认: 4)"
    )
    
    parser.add_argument(
        "--record",
        action="store_true",
        help="启用视频录制"
    )
    
    parser.add_argument(
        "--no-gui",
        action="store_true", 
        help="无GUI模式运行"
    )
    
    parser.add_argument(
        "--table-type",
        choices=["POCKET", "SNOOKER", "BILLIARD"],
        default="POCKET",
        help="台球桌类型"
    )
    
    return parser.parse_args()

def run_demo_mode(integration, shots_count):
    """运行演示模式"""
    print(f"🎯 演示模式: {shots_count}次击球演示")
    
    # 预定义的演示击球
    demo_shots = [
        {"name": "直线击球", "velocity": 4.0, "phi": 0.0, "theta": 0.0},
        {"name": "角度击球", "velocity": 5.5, "phi": 0.524, "theta": 0.0},  # 30度
        {"name": "低速控制", "velocity": 2.8, "phi": -0.262, "theta": 0.0},  # -15度
        {"name": "高速冲击", "velocity": 7.2, "phi": 0.785, "theta": 0.087},  # 45度+5度俯仰
        {"name": "精确旋转", "velocity": 3.5, "phi": 0.175, "theta": 0.0},  # 10度
        {"name": "复杂击球", "velocity": 6.0, "phi": -0.436, "theta": 0.052},  # -25度+3度俯仰
    ]
    
    # 限制击球次数
    selected_shots = demo_shots[:min(shots_count, len(demo_shots))]
    
    for i, shot in enumerate(selected_shots, 1):
        print(f"\n{'='*50}")
        print(f"第{i}击: {shot['name']}")
        print(f"{'='*50}")
        
        # 设置相机视角
        camera_views = ["overview", "robot_view", "table_view", "action_view"]
        integration.set_camera_view(camera_views[i % len(camera_views)])
        
        # 执行击球
        result = integration.execute_pool_shot(
            velocity=shot["velocity"],
            phi=shot["phi"], 
            theta=shot["theta"]
        )
        
        print(f"击球结果: {result.get('execution_quality', '未知')}")
        
        # 观察时间
        time.sleep(3)
        
        # 重置环境(最后一击除外)
        if i < len(selected_shots):
            integration.reset_simulation()
            time.sleep(1)

def run_interactive_mode(integration):
    """运行交互模式"""
    print("🎮 交互模式启动")
    print("可用命令:")
    print("  shot <速度> [角度] [俯仰] - 执行击球")
    print("  reset - 重置环境")
    print("  camera <视角> - 切换摄像头视角")
    print("  quit - 退出")
    
    while True:
        try:
            command = input("\n🎯 输入命令: ").strip().split()
            
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == "quit" or cmd == "q":
                break
            elif cmd == "shot":
                # 解析击球参数
                velocity = float(command[1]) if len(command) > 1 else 4.0
                phi = float(command[2]) if len(command) > 2 else 0.0
                theta = float(command[3]) if len(command) > 3 else 0.0
                
                print(f"执行击球: 速度={velocity}, 水平角={phi}, 俯仰角={theta}")
                
                result = integration.execute_pool_shot(
                    velocity=velocity,
                    phi=phi,
                    theta=theta
                )
                
                print(f"击球完成: {result.get('execution_quality', '未知')}")
                
            elif cmd == "reset":
                integration.reset_simulation()
                print("环境已重置")
                
            elif cmd == "camera":
                view = command[1] if len(command) > 1 else "overview"
                integration.set_camera_view(view)
                
            else:
                print(f"未知命令: {cmd}")
                
        except (ValueError, IndexError) as e:
            print(f"参数错误: {e}")
        except KeyboardInterrupt:
            break

def run_benchmark_mode(integration, shots_count):
    """运行性能测试模式"""
    print(f"⚡ 性能测试模式: {shots_count}次击球")
    
    import time
    import statistics
    
    execution_times = []
    
    for i in range(shots_count):
        print(f"测试击球 {i+1}/{shots_count}...")
        
        start_time = time.time()
        
        # 执行标准击球
        result = integration.execute_pool_shot(
            velocity=4.0 + (i % 3),  # 变化速度
            phi=(i % 7) * 0.1 - 0.3,  # 变化角度
            theta=0.0
        )
        
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        print(f"  执行时间: {execution_time:.2f}s")
        
        # 快速重置
        integration.reset_simulation()
    
    # 统计结果
    avg_time = statistics.mean(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)
    
    print(f"\n📊 性能测试结果:")
    print(f"  平均执行时间: {avg_time:.2f}s")
    print(f"  最快执行时间: {min_time:.2f}s")
    print(f"  最慢执行时间: {max_time:.2f}s")
    print(f"  总击球次数: {shots_count}")

def main():
    """主函数"""
    args = parse_arguments()
    
    print("🤖 Franka Panda + Pooltool 协同仿真系统")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"台球桌类型: {args.table_type}")
    print(f"GUI模式: {'否' if args.no_gui else '是'}")
    print(f"视频录制: {'是' if args.record else '否'}")
    print("=" * 60)
    
    try:
        # 初始化协同仿真系统
        integration = FrankaPooltoolIntegration(
            use_gui=not args.no_gui,
            enable_video_recording=args.record,
            table_type=args.table_type
        )
        
        # 根据模式运行
        if args.mode == "demo":
            run_demo_mode(integration, args.shots)
        elif args.mode == "interactive":
            run_interactive_mode(integration)
        elif args.mode == "benchmark":
            run_benchmark_mode(integration, args.shots)
        
        # 等待用户交互(GUI模式)
        if not args.no_gui and args.mode != "interactive":
            print("\n⌨️ 按回车键退出...")
            input()
        
    except KeyboardInterrupt:
        print("\n⚡ 用户中断")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if 'integration' in locals():
            integration.close()
        print("👋 再见!")

if __name__ == "__main__":
    main() 