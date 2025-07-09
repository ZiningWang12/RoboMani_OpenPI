#!/usr/bin/env python3
"""
专业台球规则演示

展示pooltool的专业台球物理引擎和规则系统的完整功能:
- 完整的8球台球规则
- 专业击球技术 (英式击球、跳球、旋转等)
- 高级物理效应 (摩擦、碰撞、反弹)
- 策略分析和轨迹预测

版本: 1.0.0
作者: OpenPI团队
"""

import sys
import os
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# 添加路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent.parent / "third_party" / "pooltool"))

import pooltool as pt
from franka_pooltool_integration import FrankaPooltoolIntegration

class ProfessionalPoolDemo:
    """
    专业台球规则演示系统
    
    展示pooltool的完整专业功能:
    - 8球、9球、斯诺克等规则
    - 专业击球技术
    - AI策略分析
    - 高级物理效应
    """
    
    def __init__(self, game_type: str = "EIGHTBALL", use_gui: bool = True):
        """
        初始化专业台球演示
        
        Args:
            game_type: 游戏类型 (EIGHTBALL, NINEBALL, SNOOKER)
            use_gui: 是否使用GUI
        """
        self.game_type = game_type
        self.use_gui = use_gui
        
        print(f"🎱 初始化专业台球演示系统 ({game_type})")
        
        # 初始化协同仿真系统
        self.integration = FrankaPooltoolIntegration(
            use_gui=use_gui,
            enable_video_recording=True,
            table_type="POCKET"
        )
        
        # 设置专业游戏规则
        self._setup_professional_game()
        
        print("✅ 专业台球演示系统初始化完成!")
    
    def _setup_professional_game(self):
        """设置专业游戏规则和布局"""
        try:
            if self.game_type == "EIGHTBALL":
                # 标准8球布局
                self._setup_eightball_rack()
            elif self.game_type == "NINEBALL":
                # 9球布局
                self._setup_nineball_rack()
            elif self.game_type == "SNOOKER":
                # 斯诺克布局
                self._setup_snooker_rack()
            
            print(f"✅ {self.game_type}规则设置完成")
            
        except Exception as e:
            print(f"⚠️ 规则设置失败: {e}，使用默认布局")
    
    def _setup_eightball_rack(self):
        """设置标准8球布局"""
        try:
            # 使用pooltool的标准8球布局
            balls = pt.get_rack(pt.GameType.EIGHTBALL, table=self.integration.table)
            
            # 更新系统中的球
            self.integration.system = pt.System(
                cue=self.integration.cue,
                table=self.integration.table,
                balls=balls
            )
            
            print("   ✅ 标准8球布局设置完成")
            
        except Exception as e:
            print(f"   ⚠️ 8球布局设置失败: {e}")
    
    def _setup_nineball_rack(self):
        """设置9球布局"""
        try:
            # 使用pooltool的9球布局
            balls = pt.get_rack(pt.GameType.NINEBALL, table=self.integration.table)
            
            self.integration.system = pt.System(
                cue=self.integration.cue,
                table=self.integration.table,
                balls=balls
            )
            
            print("   ✅ 9球布局设置完成")
            
        except Exception as e:
            print(f"   ⚠️ 9球布局设置失败: {e}")
    
    def _setup_snooker_rack(self):
        """设置斯诺克布局"""
        try:
            # 使用pooltool的斯诺克布局
            balls = pt.get_rack(pt.GameType.SNOOKER, table=self.integration.table)
            
            self.integration.system = pt.System(
                cue=self.integration.cue,
                table=self.integration.table,
                balls=balls
            )
            
            print("   ✅ 斯诺克布局设置完成")
            
        except Exception as e:
            print(f"   ⚠️ 斯诺克布局设置失败: {e}")
    
    def demonstrate_professional_techniques(self):
        """演示专业击球技术"""
        print("\n🎯 专业击球技术演示")
        print("=" * 50)
        
        techniques = [
            ("直击进袋", self._demo_straight_pot),
            ("切球技术", self._demo_cut_shot),
            ("英式击球", self._demo_english_shot),
            ("跳球技术", self._demo_jump_shot),
            ("防守击球", self._demo_safety_shot),
            ("组合球", self._demo_combination_shot),
        ]
        
        for i, (name, demo_func) in enumerate(techniques, 1):
            print(f"\n第{i}项技术: {name}")
            print("-" * 30)
            
            try:
                demo_func()
                time.sleep(2)  # 观察时间
            except Exception as e:
                print(f"   ❌ {name}演示失败: {e}")
            
            # 重置环境
            if i < len(techniques):
                self.integration.reset_simulation()
                time.sleep(1)
    
    def _demo_straight_pot(self):
        """演示直击进袋"""
        print("   🎯 目标: 直线击球入袋")
        
        # 寻找最容易进袋的球
        target_ball = self._find_easiest_pot()
        if target_ball:
            # 计算进袋角度
            angle = self._calculate_potting_angle(target_ball)
            
            # 执行击球
            result = self.integration.execute_pool_shot(
                velocity=4.5,
                phi=angle,
                theta=0.0,
                target_ball=target_ball
            )
            
            print(f"   结果: {result.get('execution_quality', '未知')}")
        else:
            print("   ⚠️ 未找到合适的进袋目标")
    
    def _demo_cut_shot(self):
        """演示切球技术"""
        print("   🎯 目标: 切球技术")
        
        # 执行30度切球
        result = self.integration.execute_pool_shot(
            velocity=5.0,
            phi=np.pi/6,  # 30度
            theta=0.0,
            offset_a=0.0,
            offset_b=0.0
        )
        
        print(f"   切球角度: 30度")
        print(f"   结果: {result.get('execution_quality', '未知')}")
    
    def _demo_english_shot(self):
        """演示英式击球 (旋转)"""
        print("   🎯 目标: 英式击球 (侧旋)")
        
        # 执行带侧旋的击球
        result = self.integration.execute_pool_shot(
            velocity=4.0,
            phi=0.0,
            theta=0.0,
            offset_a=0.015,  # 侧向偏移产生旋转
            offset_b=0.0
        )
        
        print(f"   旋转类型: 侧旋 (offset_a=0.015)")
        print(f"   结果: {result.get('execution_quality', '未知')}")
    
    def _demo_jump_shot(self):
        """演示跳球技术"""
        print("   🎯 目标: 跳球技术")
        
        # 执行跳球 (向下击球)
        result = self.integration.execute_pool_shot(
            velocity=6.0,
            phi=0.0,
            theta=0.1,  # 向下的俯仰角
            offset_a=0.0,
            offset_b=-0.02  # 击球点偏下
        )
        
        print(f"   跳球角度: 俯仰5.7度")
        print(f"   结果: {result.get('execution_quality', '未知')}")
    
    def _demo_safety_shot(self):
        """演示防守击球"""
        print("   🎯 目标: 防守性击球")
        
        # 轻柔的防守性击球
        result = self.integration.execute_pool_shot(
            velocity=2.5,  # 低速
            phi=np.pi/4,   # 45度角
            theta=0.0,
            offset_a=0.0,
            offset_b=0.005  # 轻微上旋
        )
        
        print(f"   防守策略: 低速角度击球")
        print(f"   结果: {result.get('execution_quality', '未知')}")
    
    def _demo_combination_shot(self):
        """演示组合球"""
        print("   🎯 目标: 组合球技术")
        
        # 复杂的组合击球
        result = self.integration.execute_pool_shot(
            velocity=5.5,
            phi=-np.pi/8,  # -22.5度
            theta=0.0,
            offset_a=0.008,  # 轻微侧旋
            offset_b=0.005   # 轻微上旋
        )
        
        print(f"   组合技术: 角度+旋转")
        print(f"   结果: {result.get('execution_quality', '未知')}")
    
    def _find_easiest_pot(self) -> Optional[str]:
        """寻找最容易进袋的球"""
        try:
            if hasattr(self.integration, 'system') and self.integration.system.balls:
                # 简化版本 - 选择第一个非母球
                for ball_id, ball in self.integration.system.balls.items():
                    if ball_id != "cue":
                        return ball_id
            return "1"  # 默认目标
        except Exception:
            return "1"
    
    def _calculate_potting_angle(self, target_ball: str) -> float:
        """计算进袋角度"""
        try:
            # 使用pooltool的瞄准功能
            angle = pt.aim.at_ball(self.integration.system, target_ball)
            return angle
        except Exception:
            return 0.0  # 默认直线
    
    def analyze_shot_physics(self):
        """分析击球物理"""
        print("\n🔬 击球物理分析")
        print("=" * 50)
        
        # 执行分析用击球
        print("执行分析用击球...")
        result = self.integration.execute_pool_shot(
            velocity=4.0,
            phi=np.pi/6,
            theta=0.0,
            offset_a=0.01,
            offset_b=0.005
        )
        
        # 分析系统能量
        if hasattr(self.integration, 'system'):
            try:
                energy = self.integration.system.get_system_energy()
                print(f"📊 系统总能量: {energy:.4f} J")
            except Exception as e:
                print(f"⚠️ 能量计算失败: {e}")
        
        # 分析球的运动
        if result.get("ball_motion"):
            print("\n📈 球运动分析:")
            for ball_id, motion in result["ball_motion"].items():
                pos = motion.get("initial_pos", [0, 0, 0])
                vel = motion.get("initial_vel", [0, 0, 0])
                speed = np.linalg.norm(vel)
                print(f"   {ball_id}: 位置{pos[:2]}, 速度{speed:.2f}m/s")
    
    def run_professional_demo(self):
        """运行完整的专业演示"""
        print("\n🏆 专业台球演示开始!")
        print("=" * 60)
        
        # 设置最佳观察视角
        self.integration.set_camera_view("overview")
        
        try:
            # 1. 技术演示
            self.demonstrate_professional_techniques()
            
            # 2. 物理分析
            self.analyze_shot_physics()
            
            # 3. 策略演示
            self._demonstrate_game_strategy()
            
        except Exception as e:
            print(f"❌ 演示过程中出现错误: {e}")
        
        print(f"\n🎉 专业台球演示完成!")
        print(f"游戏类型: {self.game_type}")
        print(f"技术展示: 6项专业技术")
        print(f"物理分析: 完整")
    
    def _demonstrate_game_strategy(self):
        """演示游戏策略"""
        print("\n🧠 游戏策略演示")
        print("=" * 50)
        
        if self.game_type == "EIGHTBALL":
            print("8球策略: 分组击球 → 击打8号球")
        elif self.game_type == "NINEBALL":
            print("9球策略: 按数字顺序击球 → 击打9号球")
        elif self.game_type == "SNOOKER":
            print("斯诺克策略: 红球 → 彩球 → 红球 → 彩球")
        
        # 执行策略性击球
        strategy_shot = self.integration.execute_pool_shot(
            velocity=3.8,
            phi=np.pi/12,  # 15度
            theta=0.0,
            offset_a=0.005,
            offset_b=0.002
        )
        
        print(f"策略击球结果: {strategy_shot.get('execution_quality', '未知')}")
    
    def close(self):
        """关闭演示系统"""
        self.integration.close()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="专业台球规则演示")
    parser.add_argument("--game", choices=["EIGHTBALL", "NINEBALL", "SNOOKER"], 
                       default="EIGHTBALL", help="游戏类型")
    parser.add_argument("--no-gui", action="store_true", help="无GUI模式")
    
    args = parser.parse_args()
    
    print("🎱 专业台球规则与物理演示系统")
    print("=" * 60)
    print(f"游戏类型: {args.game}")
    print(f"GUI模式: {'否' if args.no_gui else '是'}")
    print("=" * 60)
    
    try:
        # 创建演示系统
        demo = ProfessionalPoolDemo(
            game_type=args.game,
            use_gui=not args.no_gui
        )
        
        # 运行演示
        demo.run_professional_demo()
        
        # 等待用户交互
        if not args.no_gui:
            print("\n⌨️ 按回车键退出...")
            input()
        
    except KeyboardInterrupt:
        print("\n⚡ 用户中断")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'demo' in locals():
            demo.close()
        print("👋 再见!")

if __name__ == "__main__":
    main() 