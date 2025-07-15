#!/usr/bin/env python3
"""
Milestone 2: PoolTool + Mujoco机械臂集成
目标: 集成PoolTool与Mujoco机器人仿真，生成机器人+台球的同环境仿真demo
"""

import sys
import os
import numpy as np
import pathlib
import json
from typing import Optional, List, Dict, Any
import traceback

# 导入Mujoco相关
import mujoco as mj
import mujoco.viewer

# 确保可以导入pooltool
sys.path.insert(0, '/app/third_party/pooltool')

# 设置环境变量
os.environ['PANDA3D_WINDOW_TYPE'] = 'offscreen'

def test_imports():
    """测试所有必要模块的导入"""
    print("=== 测试模块导入 ===")
    
    try:
        # 测试PoolTool
        import pooltool as pt
        print(f"✅ PoolTool版本: {pt.__version__}")
        
        # 测试Mujoco
        print(f"✅ Mujoco版本: {mj.__version__}")
        
        # 测试LIBERO（参考机械臂模型）
        # 设置环境变量避免交互式输入
        os.environ['LIBERO_DATASET_PATH'] = '/app/third_party/libero/datasets'
        sys.path.insert(0, '/app/third_party/libero')
        try:
            from libero.libero.envs import OffScreenRenderEnv
            print("✅ LIBERO环境可导入")
        except Exception as libero_e:
            print(f"⚠️ LIBERO导入失败(可忽略): {libero_e}")
            # 继续执行，不影响主要功能
        
        return True
        
    except Exception as e:
        print(f"❌ 导入测试失败: {e}")
        traceback.print_exc()
        return False

def create_mujoco_table_scene():
    """创建Mujoco中的台球桌场景"""
    print("\n=== 创建Mujoco台球桌场景 ===")
    
    # 创建基础的MJCModel XML
    xml_content = """
    <mujoco model="pool_table">
        <option timestep="0.01" gravity="0 0 -9.81"/>
        
        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
        </visual>
        
        <asset>
            <!-- 台球桌材质 -->
            <material name="table_felt" rgba="0.1 0.6 0.1 1"/>
            <material name="table_wood" rgba="0.6 0.3 0.1 1"/>
            <material name="ball_white" rgba="1 1 1 1"/>
            <material name="ball_yellow" rgba="1 1 0 1"/>
            <material name="ball_red" rgba="1 0 0 1"/>
        </asset>
        
        <worldbody>
            <!-- 地面 -->
            <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
            
            <!-- 台球桌 -->
            <body name="table" pos="0 0 0.4">
                <!-- 桌面 -->
                <geom name="table_surface" type="box" size="0.9906 1.9812 0.036" 
                      material="table_felt" pos="0 0 0"/>
                      
                <!-- 桌边 -->
                <geom name="table_edge_1" type="box" size="1.0 0.05 0.1" 
                      material="table_wood" pos="0 2.03 0.1"/>
                <geom name="table_edge_2" type="box" size="1.0 0.05 0.1" 
                      material="table_wood" pos="0 -2.03 0.1"/>
                <geom name="table_edge_3" type="box" size="0.05 1.98 0.1" 
                      material="table_wood" pos="1.04 0 0.1"/>
                <geom name="table_edge_4" type="box" size="0.05 1.98 0.1" 
                      material="table_wood" pos="-1.04 0 0.1"/>
            </body>
            
            <!-- 示例台球 -->
            <body name="cue_ball" pos="0 -0.5 0.464">
                <joint name="cue_ball_joint" type="free"/>
                <geom name="cue_ball_geom" type="sphere" size="0.028575" 
                      material="ball_white" mass="0.17"/>
            </body>
            
            <body name="ball_1" pos="0 0.5 0.464">
                <joint name="ball_1_joint" type="free"/>
                <geom name="ball_1_geom" type="sphere" size="0.028575" 
                      material="ball_yellow" mass="0.17"/>
            </body>
        </worldbody>
    </mujoco>
    """
    
    # 保存XML文件
    xml_path = "/app/data/pooltool/milestone2/pool_table.xml"
    pathlib.Path(xml_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(xml_path, 'w') as f:
        f.write(xml_content)
    
    print(f"✅ 台球桌XML模型保存到: {xml_path}")
    
    try:
        # 加载模型
        model = mj.MjModel.from_xml_path(xml_path)
        data = mj.MjData(model)
        
        print(f"✅ Mujoco模型加载成功")
        print(f"   - 物体数量: {model.nbody}")
        print(f"   - 关节数量: {model.njnt}")
        print(f"   - 几何体数量: {model.ngeom}")
        
        return model, data, xml_path
        
    except Exception as e:
        print(f"❌ Mujoco模型加载失败: {e}")
        return None, None, xml_path

def create_franka_arm_scene():
    """创建带有Franka机械臂的场景"""
    print("\n=== 集成Franka机械臂 ===")
    
    try:
        # 参考LIBERO的Franka机械臂配置
        # 这里我们创建一个简化版本
        franka_xml = """
        <mujoco model="franka_pool">
            <option timestep="0.01" gravity="0 0 -9.81"/>
            
            <visual>
                <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3"/>
                <global offwidth="1280" offheight="720"/>
            </visual>
            
            <asset>
                <material name="table_felt" rgba="0.1 0.6 0.1 1"/>
                <material name="table_wood" rgba="0.6 0.3 0.1 1"/>
                <material name="ball_white" rgba="1 1 1 1"/>
                <material name="ball_yellow" rgba="1 1 0 1"/>
                <material name="robot_grey" rgba="0.7 0.7 0.7 1"/>
            </asset>
            
            <worldbody>
                <!-- 地面 -->
                <geom name="floor" type="plane" size="3 3 0.1" rgba="0.8 0.8 0.8 1"/>
                
                <!-- 台球桌 -->
                <body name="table" pos="0 0 0.4">
                    <geom name="table_surface" type="box" size="0.9906 1.9812 0.036" 
                          material="table_felt" pos="0 0 0"/>
                    <geom name="table_edge_1" type="box" size="1.0 0.05 0.1" 
                          material="table_wood" pos="0 2.03 0.1"/>
                    <geom name="table_edge_2" type="box" size="1.0 0.05 0.1" 
                          material="table_wood" pos="0 -2.03 0.1"/>
                    <geom name="table_edge_3" type="box" size="0.05 1.98 0.1" 
                          material="table_wood" pos="1.04 0 0.1"/>
                    <geom name="table_edge_4" type="box" size="0.05 1.98 0.1" 
                          material="table_wood" pos="-1.04 0 0.1"/>
                </body>
                
                <!-- 简化版Franka机械臂 -->
                <body name="franka_base" pos="-1.5 0 0.4">
                    <!-- 基座 -->
                    <geom name="base" type="cylinder" size="0.08 0.1" material="robot_grey"/>
                    
                    <!-- 关节1 -->
                    <body name="link1" pos="0 0 0.1">
                        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                        <geom name="link1_geom" type="cylinder" size="0.06 0.1" material="robot_grey"/>
                        
                        <!-- 关节2 -->
                        <body name="link2" pos="0 0 0.1">
                            <joint name="joint2" type="hinge" axis="0 1 0" range="-1.7628 1.7628"/>
                            <geom name="link2_geom" type="box" size="0.05 0.05 0.15" material="robot_grey"/>
                            
                            <!-- 关节3 -->
                            <body name="link3" pos="0 0 0.15">
                                <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                                <geom name="link3_geom" type="cylinder" size="0.05 0.1" material="robot_grey"/>
                                
                                <!-- 末端执行器 -->
                                <body name="end_effector" pos="0 0 0.1">
                                    <joint name="joint4" type="hinge" axis="0 1 0" range="-3.0718 -0.0698"/>
                                    <geom name="ee_geom" type="box" size="0.03 0.03 0.08" material="robot_grey"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
                
                <!-- 台球 -->
                <body name="cue_ball" pos="0 -0.5 0.464">
                    <joint name="cue_ball_joint" type="free"/>
                    <geom name="cue_ball_geom" type="sphere" size="0.028575" 
                          material="ball_white" mass="0.17"/>
                </body>
                
                <body name="ball_1" pos="0 0.5 0.464">
                    <joint name="ball_1_joint" type="free"/>
                    <geom name="ball_1_geom" type="sphere" size="0.028575" 
                          material="ball_yellow" mass="0.17"/>
                </body>
            </worldbody>
            
            <actuator>
                <!-- 关节执行器 -->
                <motor name="motor1" joint="joint1" gear="100"/>
                <motor name="motor2" joint="joint2" gear="100"/>
                <motor name="motor3" joint="joint3" gear="100"/>
                <motor name="motor4" joint="joint4" gear="100"/>
            </actuator>
        </mujoco>
        """
        
        # 保存XML文件
        xml_path = "/app/data/pooltool/milestone2/franka_pool.xml"
        with open(xml_path, 'w') as f:
            f.write(franka_xml)
        
        print(f"✅ Franka+台球XML模型保存到: {xml_path}")
        
        # 加载模型
        model = mj.MjModel.from_xml_path(xml_path)
        data = mj.MjData(model)
        
        print(f"✅ Franka+台球模型加载成功")
        print(f"   - 物体数量: {model.nbody}")
        print(f"   - 关节数量: {model.njnt}")
        print(f"   - 执行器数量: {model.nu}")
        
        return model, data, xml_path
        
    except Exception as e:
        print(f"❌ Franka模型创建失败: {e}")
        traceback.print_exc()
        return None, None, None

def run_mujoco_simulation(model, data, duration: float = 5.0):
    """运行Mujoco仿真"""
    print(f"\n=== 运行Mujoco仿真 ({duration}秒) ===")
    
    try:
        # 初始化
        mj.mj_resetData(model, data)
        
        # 设置初始状态
        if model.nu > 0:  # 如果有执行器
            # 简单的正弦波运动
            data.ctrl[:] = 0.0
        
        # 仿真循环
        simulation_data = []
        steps = int(duration / model.opt.timestep)
        
        for step in range(steps):
            # 简单的控制逻辑（示例）
            if model.nu > 0:
                time = step * model.opt.timestep
                data.ctrl[0] = 0.5 * np.sin(time)  # 第一个关节正弦运动
            
            # 步进仿真
            mj.mj_step(model, data)
            
            # 记录数据
            if step % 10 == 0:  # 每10步记录一次
                frame_data = {
                    "time": data.time,
                    "qpos": data.qpos.copy().tolist(),
                    "qvel": data.qvel.copy().tolist(),
                    "ctrl": data.ctrl.copy().tolist() if model.nu > 0 else []
                }
                simulation_data.append(frame_data)
        
        print(f"✅ 仿真完成，运行了{steps}步")
        print(f"✅ 记录了{len(simulation_data)}帧数据")
        
        return simulation_data
        
    except Exception as e:
        print(f"❌ 仿真运行失败: {e}")
        traceback.print_exc()
        return []

def integrate_pooltool_mujoco():
    """集成PoolTool物理引擎与Mujoco可视化"""
    print("\n=== 集成PoolTool与Mujoco ===")
    
    try:
        # 1. 使用PoolTool计算台球物理
        import pooltool as pt
        
        # 创建PoolTool场景
        table = pt.Table.default()
        balls = pt.get_rack(pt.GameType.NINEBALL, table, spacing_factor=1e-3)
        shot = pt.System(
            cue=pt.Cue(cue_ball_id="cue"),
            table=table,
            balls=balls,
        )
        shot.strike(V0=6.0, phi=pt.aim.at_ball(shot, "1"))
        
        # 运行PoolTool物理仿真
        print("运行PoolTool物理仿真...")
        pt.simulate(shot, inplace=True)
        pt.continuize(shot, inplace=True)
        
        print("✅ PoolTool物理仿真完成")
        
        # 2. 提取PoolTool数据用于Mujoco可视化
        ball_trajectories = {}
        for ball_id, ball in shot.balls.items():
            trajectory = ball.state.rvw  # 位置、速度、角速度
            ball_trajectories[ball_id] = {
                "positions": trajectory[:, 0:3],  # x, y, z
                "velocities": trajectory[:, 3:6],  # vx, vy, vz
                "num_frames": len(trajectory)
            }
        
        print(f"✅ 提取了{len(ball_trajectories)}个球的轨迹数据")
        
        # 3. 准备Mujoco可视化数据
        integration_data = {
            "pooltool_table": {
                "length": float(table.l),
                "width": float(table.w),
                "height": float(table.height)
            },
            "ball_trajectories": {
                ball_id: {
                    "positions": traj["positions"].tolist(),
                    "velocities": traj["velocities"].tolist(),
                    "num_frames": traj["num_frames"]
                }
                for ball_id, traj in ball_trajectories.items()
            }
        }
        
        # 保存集成数据
        integration_path = "/app/data/pooltool/milestone2/pooltool_mujoco_integration.json"
        with open(integration_path, 'w') as f:
            json.dump(integration_data, f, indent=2)
        
        print(f"✅ 集成数据保存到: {integration_path}")
        return integration_data
        
    except Exception as e:
        print(f"❌ PoolTool-Mujoco集成失败: {e}")
        traceback.print_exc()
        return None

def main():
    """主函数：运行Milestone 2集成测试"""
    print("=== PoolTool x OpenPI Milestone 2: Mujoco机械臂集成 ===")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 1. 测试导入
    if not test_imports():
        print("❌ 导入测试失败，无法继续")
        return
    
    # 2. 创建输出目录
    output_dir = pathlib.Path("/app/data/pooltool/milestone2")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    try:
        # 3. 创建基础台球桌场景
        table_model, table_data, table_xml = create_mujoco_table_scene()
        
        if table_model is not None:
            # 运行基础仿真
            table_sim_data = run_mujoco_simulation(table_model, table_data, duration=3.0)
            
            # 保存仿真数据
            table_sim_path = output_dir / "table_simulation.json"
            with open(table_sim_path, 'w') as f:
                json.dump(table_sim_data, f, indent=2)
            print(f"✅ 台球桌仿真数据保存到: {table_sim_path}")
        
        # 4. 创建Franka机械臂场景
        franka_model, franka_data, franka_xml = create_franka_arm_scene()
        
        if franka_model is not None:
            # 运行Franka仿真
            franka_sim_data = run_mujoco_simulation(franka_model, franka_data, duration=5.0)
            
            # 保存仿真数据
            franka_sim_path = output_dir / "franka_simulation.json"
            with open(franka_sim_path, 'w') as f:
                json.dump(franka_sim_data, f, indent=2)
            print(f"✅ Franka仿真数据保存到: {franka_sim_path}")
        
        # 5. 集成PoolTool与Mujoco
        integration_data = integrate_pooltool_mujoco()
        
        print(f"\n🎉 Milestone 2 集成测试完成！")
        print(f"生成的文件保存在: {output_dir}")
        print("\n生成的文件列表:")
        for file_path in sorted(output_dir.glob("*")):
            print(f"  - {file_path.name}")
        
    except Exception as e:
        print(f"❌ Milestone 2 过程中出错: {e}")
        traceback.print_exc()
    
    print("\n下一步: Milestone 3 - 实现机械臂与台球的物理交互")

if __name__ == "__main__":
    main() 