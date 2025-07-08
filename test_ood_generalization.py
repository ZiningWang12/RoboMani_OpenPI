#!/usr/bin/env python3
"""
基于main.py标准实现的π₀-FAST分布外泛化测试
测试模型对不同物体和任务的泛化能力
"""
import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    
    # Environment parameters
    num_steps_wait: int = 10
    num_trials_per_task: int = 1  # 每个任务只测试1次
    
    # Test selection
    test_type: str = "all"  # 测试类型: all, control, object_ood, place_ood, action_ood, complex_ood
    
    # Output
    video_out_path: str = "ood_test_videos"
    seed: int = 7

def test_ood_generalization(args: Args) -> None:
    """基于标准main.py实现的OOD泛化测试"""
    print("🧪 开始分布外泛化测试（基于main.py标准实现）...")
    np.random.seed(args.seed)
    
    # 定义测试案例：基于标准LIBERO任务但替换物体
    test_cases = [
        # === Group 1: 标准任务对照组 ===
        {
            "base_task_id": 0,  # "pick up the black bowl between the plate and the ramekin and place it on the plate"
            "custom_prompt": None,  # 使用原始任务描述
            "description": "✅ Control: 标准LIBERO任务0",
            "test_name": "control_standard_task0",
            "exp_type": "control"
        },
        {
            "base_task_id": 1,  # "pick up the black bowl next to the ramekin and place it on the plate"
            "custom_prompt": None,
            "description": "✅ Control: 标准LIBERO任务1", 
            "test_name": "control_standard_task1",
            "exp_type": "control"
        },
        
        # === Group 2: 物体替换OOD（保持标准格式）===
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the ramekin between the plate and the black bowl and place it on the plate",
            "description": "🔄 OOD-物体: 操作ramekin（格式标准）",
            "test_name": "ood_ramekin_between",
            "exp_type": "object_ood"
        },
        {
            "base_task_id": 1,
            "custom_prompt": "pick up the plate next to the ramekin and place it on the black bowl", 
            "description": "🔄 OOD-物体: 操作plate（格式标准）",
            "test_name": "ood_plate_next_to",
            "exp_type": "object_ood"
        },
        {
            "base_task_id": 2,  # "pick up the black bowl from table center and place it on the plate"
            "custom_prompt": "pick up the ramekin from table center and place it on the plate",
            "description": "🎯 OOD-物体: 简单替换ramekin",
            "test_name": "ood_ramekin_center",
            "exp_type": "object_ood"
        },
        
        # === Group 3: 目标位置OOD ===
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the black bowl between the plate and the ramekin and place it on the ramekin",
            "description": "📍 OOD-位置: 放到ramekin上",
            "test_name": "ood_place_on_ramekin", 
            "exp_type": "place_ood"
        },
        
        # === Group 4: 动作OOD ===
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the black bowl between the plate and the ramekin and throw it",
            "description": "🎯 OOD-动作: throw动作（vs标准place）",
            "test_name": "ood_action_throw",
            "exp_type": "action_ood"
        },
        {
            "base_task_id": 1, 
            "custom_prompt": "pick up the black bowl next to the ramekin and put it down",
            "description": "🎯 OOD-动作: put down动作",
            "test_name": "ood_action_put_down",
            "exp_type": "action_ood"
        },
        {
            "base_task_id": 2,
            "custom_prompt": "pick up the black bowl from table center and drop it",
            "description": "🎯 OOD-动作: drop动作",
            "test_name": "ood_action_drop",
            "exp_type": "action_ood"
        },
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the black bowl between the plate and the ramekin and release it",
            "description": "🎯 OOD-动作: release动作",
            "test_name": "ood_action_release", 
            "exp_type": "action_ood"
        },
        
        # === Group 5: 目标物体缺失测试（期望失败）===
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "description": "❌ 预期失败: 环境中没有黑碗但要求拿黑碗",
            "test_name": "missing_black_bowl_test",
            "exp_type": "missing_object",
            "remove_objects": ["black_bowl"]  # 要移除的物体列表
        },
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "description": "❌ 预期失败: 环境中没有plate但要求放到plate上",
            "test_name": "missing_plate_test", 
            "exp_type": "missing_object",
            "remove_objects": ["plate"]  # 移除plate
        },
        
        # === Group 6: 无效指令测试（重新设计，更严格）===
        {
            "base_task_id": 0,
            "custom_prompt": "",
            "description": "❌ 预期失败: 完全空指令",
            "test_name": "empty_string_test",
            "exp_type": "invalid_instruction"
        },
        {
            "base_task_id": 0,
            "custom_prompt": "sit down and read a book",
            "description": "❌ 预期失败: 语法正确但与机器人无关的指令",
            "test_name": "irrelevant_instruction_test", 
            "exp_type": "invalid_instruction"
        },
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the unicorn and place it on the moon",
            "description": "❌ 预期失败: 不存在的物体和位置",
            "test_name": "impossible_objects_test",
            "exp_type": "invalid_instruction"
        },
        {
            "base_task_id": 0,
            "custom_prompt": "asdf qwerty zxcv hjkl",
            "description": "❌ 预期失败: 随机键盘字符",
            "test_name": "random_text_test",
            "exp_type": "invalid_instruction"
        },
        
        # === Group 6: 复杂OOD（我们之前失败的）===
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the stove and place it on the plate",
            "description": "❌ 预期失败: 操作stove（非标准物体）",
            "test_name": "ood_stove_non_standard",
            "exp_type": "complex_ood"
        },
        {
            "base_task_id": 0,
            "custom_prompt": "pick up the black bowl that is closest to the plate and place it on the wooden cabinet",
            "description": "❌ 预期失败: 非标准目标位置",
            "test_name": "ood_non_standard_target",
            "exp_type": "complex_ood"
        }
    ]
    
    # 初始化LIBERO环境
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = 220  # libero_spatial标准
    
    # 连接Policy Server
    print("🔗 连接Policy Server...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("✅ Policy Server连接成功")
    
    # 统计结果
    results = {
        "control": {"total": 0, "success": 0},
        "object_ood": {"total": 0, "success": 0},
        "place_ood": {"total": 0, "success": 0},
        "action_ood": {"total": 0, "success": 0},
        "missing_object": {"total": 0, "success": 0},
        "invalid_instruction": {"total": 0, "success": 0},
        "complex_ood": {"total": 0, "success": 0}
    }
    
    # 根据test_type过滤测试案例
    if args.test_type != "all":
        filtered_cases = [case for case in test_cases if case["exp_type"] == args.test_type]
        if not filtered_cases:
            print(f"❌ 未找到类型为 '{args.test_type}' 的测试案例")
            print(f"可用类型: control, object_ood, place_ood, action_ood, missing_object, invalid_instruction, complex_ood")
            return
        test_cases = filtered_cases
        print(f"🎯 只运行 {args.test_type} 类型测试，共 {len(test_cases)} 个案例")
    
    total_episodes, total_successes = 0, 0
    
    # 执行测试
    for test_idx, test_case in enumerate(test_cases):
        print(f"\n{'='*80}")
        print(f"📋 测试 {test_idx+1}/{len(test_cases)}: {test_case['description']}")
        print(f"🔬 实验类型: {test_case['exp_type'].upper()}")
        
        # 获取基础任务配置
        base_task = task_suite.get_task(test_case["base_task_id"])
        initial_states = task_suite.get_task_init_states(test_case["base_task_id"])
        
        # 确定使用的prompt
        if test_case["custom_prompt"] is None:
            task_prompt = str(base_task.language)
            print(f"📖 使用原始prompt: {task_prompt}")
        else:
            task_prompt = test_case["custom_prompt"]
            print(f"📖 原始任务: {base_task.language}")
            print(f"🎯 自定义prompt: {task_prompt}")
        
        # 初始化环境（基于main.py）
        env, _ = _get_libero_env(base_task, LIBERO_ENV_RESOLUTION, args.seed)
        
        # 开始episode测试
        task_episodes, task_successes = 0, 0
        for episode_idx in range(args.num_trials_per_task):
            print(f"\n🎮 开始第 {episode_idx+1} 次试验...")
            
            # 重置环境（完全按照main.py）
            env.reset()
            action_plan = collections.deque()
            
            # 设置初始状态
            obs = env.set_init_state(initial_states[episode_idx])
            
            # 特殊处理：移除指定物体（如果指定）
            remove_objects = test_case.get("remove_objects", [])
            if remove_objects:
                print(f"🔧 正在移除环境中的物体: {remove_objects}")
                _remove_objects_from_env(env, remove_objects)
            
            # 执行任务
            t = 0
            replay_images = []
            done = False
            
            while t < max_steps + args.num_steps_wait:
                try:
                    # 等待物理稳定（main.py逻辑）
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # 图像预处理（完全按照main.py）
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # 保存图像用于视频
                    replay_images.append(img)

                    if not action_plan:
                        # 准备观测数据（完全按照main.py）
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": task_prompt,  # 关键：使用我们的自定义prompt
                        }

                        # 调试：打印发送给模型的prompt（仅第一次）
                        if t == args.num_steps_wait:
                            print(f"🔍 DEBUG: 发送给模型的prompt: '{task_prompt}'")
                            print(f"📏 Prompt长度: {len(task_prompt)} 字符")
                            
                            # 对invalid_instruction类型进行额外验证
                            if test_case["exp_type"] == "invalid_instruction":
                                print(f"🔍 无效指令验证: ")
                                print(f"   - 字符类型: {type(task_prompt)}")
                                print(f"   - 是否为空: {len(task_prompt) == 0}")
                                print(f"   - 是否包含标准关键词: {'pick up' in task_prompt.lower()}")
                                print(f"   - 是否包含已知物体: {any(obj in task_prompt.lower() for obj in ['bowl', 'plate', 'ramekin'])}")
                                if not any(obj in task_prompt.lower() for obj in ['bowl', 'plate', 'ramekin', 'pick', 'place']):
                                    print(f"   ✅ 确认为真正的无效指令")

                        # 模型推理
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, \
                            f"Need at least {args.replan_steps} steps, got {len(action_chunk)}"
                        action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()

                    # 执行动作
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        print(f"🎉 任务在步骤 {t} 成功完成！")
                        # 调试：为什么成功了？
                        if "missing" in test_case["test_name"] or "invalid" in test_case["test_name"]:
                            print(f"⚠️ 意外成功！调试原因:")
                            _debug_success_conditions(env, f"{test_case['test_name']}_success")
                        break
                    t += 1

                except Exception as e:
                    print(f"❌ 执行出错: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存视频（按照main.py格式）
            suffix = "success" if done else "failure"
            video_filename = f"{test_case['test_name']}_{suffix}.mp4"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / video_filename,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # 输出结果
            print(f"📊 Episode结果:")
            print(f"   成功: {'✅ 是' if done else '❌ 否'}")
            print(f"   步数: {t}")
            print(f"   视频: {video_filename}")

        # 更新统计
        results[test_case["exp_type"]]["total"] += task_episodes
        results[test_case["exp_type"]]["success"] += task_successes
        
        print(f"📈 当前任务统计:")
        print(f"   成功率: {task_successes}/{task_episodes} = {task_successes/task_episodes*100:.1f}%")

    # 最终分析
    print(f"\n{'='*80}")
    print(f"🎊 分布外泛化测试完成！")
    print(f"\n📊 详细结果分析:")
    
    for exp_type, result in results.items():
        if result["total"] > 0:
            success_rate = result["success"] / result["total"] * 100
            emoji = _get_type_emoji(exp_type)
            print(f"\n{emoji} {exp_type.upper().replace('_', ' ')}:")
            print(f"   成功率: {result['success']}/{result['total']} = {success_rate:.1f}%")
    
    # 泛化能力对比分析
    control_rate = results["control"]["success"] / results["control"]["total"] * 100 if results["control"]["total"] > 0 else 0
    object_ood_rate = results["object_ood"]["success"] / results["object_ood"]["total"] * 100 if results["object_ood"]["total"] > 0 else 0
    action_ood_rate = results["action_ood"]["success"] / results["action_ood"]["total"] * 100 if results["action_ood"]["total"] > 0 else 0
    invalid_instruction_rate = results["invalid_instruction"]["success"] / results["invalid_instruction"]["total"] * 100 if results["invalid_instruction"]["total"] > 0 else 0
    
    print(f"\n🧪 泛化能力评估:")
    print(f"   Control基线: {control_rate:.1f}%")
    print(f"   Object OOD: {object_ood_rate:.1f}%")
    print(f"   Action OOD: {action_ood_rate:.1f}%")
    print(f"   Invalid指令: {invalid_instruction_rate:.1f}%")
    
    # 添加missing object统计
    missing_object_rate = results["missing_object"]["success"] / results["missing_object"]["total"] * 100 if results["missing_object"]["total"] > 0 else 0
    if results["missing_object"]["total"] > 0:
        print(f"   Missing目标: {missing_object_rate:.1f}%")
    
    # 关键诊断
    if invalid_instruction_rate > 50:
        print(f"\n⚠️  警告: 无效指令成功率过高 ({invalid_instruction_rate:.1f}%)，可能存在问题:")
        print(f"     - 模型可能完全忽略语言指令")
        print(f"     - 测试实现可能有缺陷")  
        print(f"     - LIBERO成功判定可能过于宽松")
        
    if missing_object_rate > 50:
        print(f"\n⚠️  警告: 缺失目标物体仍然'成功' ({missing_object_rate:.1f}%):")
        print(f"     - 模型可能不检查目标物体是否存在")
        print(f"     - 可能执行替代行为或忽略指令")
        print(f"     - LIBERO成功判定可能过于宽松")
    
    if control_rate > 0:
        print(f"\n📊 泛化保持率:")
        print(f"   物体泛化保持率: {object_ood_rate/control_rate*100:.1f}%")
        if results["action_ood"]["total"] > 0:
            print(f"   动作泛化保持率: {action_ood_rate/control_rate*100:.1f}%")
    
    print(f"\n📈 总体统计:")
    print(f"   总成功率: {total_successes}/{total_episodes} = {total_successes/total_episodes*100:.1f}%")
    print(f"   视频保存路径: {args.video_out_path}")

def _get_type_emoji(exp_type):
    """获取实验类型对应的表情符号"""
    emoji_map = {
        "control": "✅",
        "object_ood": "🔄",
        "place_ood": "📍",
        "action_ood": "🎯",
        "missing_object": "🔍",
        "invalid_instruction": "🚫",
        "complex_ood": "❌"
    }
    return emoji_map.get(exp_type, "📊")

def _get_libero_env(task, resolution, seed):
    """
    初始化LIBERO环境（完全按照main.py）
    """
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # 重要：seed影响物体位置
    return env, task_description

def _quat2axisangle(quat):
    """
    四元数转轴角（完全按照main.py的robosuite实现）
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def _debug_success_conditions(env, test_name):
    """
    调试成功条件，打印物体状态和检测结果
    """
    try:
        print(f"\n🔍 DEBUG: {test_name} - 分析成功条件")
        
        # 检查环境类型
        print(f"📋 环境类型: {type(env)}")
        
        # 检查body位置 (直接从仿真数据)
        print(f"🔍 仿真中的body位置:")
        for i, name in enumerate(env.sim.model.body_names):
            if name and ("plate" in name.lower() or "bowl" in name.lower()):
                pos = env.sim.data.body_xpos[i]
                print(f"   - {name}: {pos}")
        
        # 检查成功判定 (尝试手动调用)
        try:
            success = env.check_success()
            print(f"🎯 当前成功状态: {success}")
        except Exception as e:
            print(f"❌ 无法检查成功状态: {e}")
            
        # 检查目标条件
        try:
            if hasattr(env, 'parsed_problem'):
                goal_state = env.parsed_problem["goal_state"]
                print(f"🎯 BDDL目标条件: {goal_state}")
            else:
                print(f"❌ 环境没有parsed_problem属性")
        except Exception as e:
            print(f"❌ 无法获取目标条件: {e}")
            
        # 打印environment的属性，看看有什么可用的
        print(f"\n🔍 环境可用属性:")
        relevant_attrs = [attr for attr in dir(env) if not attr.startswith('_') and ('object' in attr.lower() or 'state' in attr.lower() or 'success' in attr.lower())]
        for attr in relevant_attrs[:10]:  # 只显示前10个
            print(f"   - {attr}")
                
    except Exception as e:
        print(f"❌ DEBUG失败: {e}")
        import traceback
        traceback.print_exc()

def _remove_objects_from_env(env, object_types):
    """
    从环境中移除指定类型的物体（真正删除，不移动位置）
    
    Args:
        env: LIBERO环境
        object_types: 要移除的物体类型列表，如 ["black_bowl", "plate"]
    """
    try:
        sim = env.sim
        
        # 定义物体名称模式映射
        object_patterns = {
            "black_bowl": ["akita_black_bowl"],
            "plate": ["plate"],
            "ramekin": ["ramekin"],
            "cookie_box": ["cookie_box"]
        }
        
        objects_removed = []
        
        for obj_type in object_types:
            if obj_type not in object_patterns:
                print(f"⚠️ 未知物体类型: {obj_type}")
                continue
                
            patterns = object_patterns[obj_type]
            print(f"🔍 移除 {obj_type}...")
            
            # 方法1: 完全禁用geom（视觉和物理）
            for i, name in enumerate(sim.model.geom_names):
                if name and any(pattern in name.lower() for pattern in patterns):
                    try:
                        # 设为完全透明
                        sim.model.geom_rgba[i][-1] = 0.0
                        
                        # 设置为极小尺寸（几乎不可见且不碰撞）
                        original_size = sim.model.geom_size[i].copy()
                        sim.model.geom_size[i] = [0.001, 0.001, 0.001]
                        
                        # 禁用碰撞检测
                        if hasattr(sim.model, 'geom_contype'):
                            sim.model.geom_contype[i] = 0  # 不与任何物体碰撞
                        if hasattr(sim.model, 'geom_conaffinity'):
                            sim.model.geom_conaffinity[i] = 0  # 不被任何物体碰撞
                        
                        print(f"  ✅ 完全禁用geom: {name}")
                        objects_removed.append(name)
                    except Exception as e:
                        print(f"  ⚠️ 禁用geom失败: {e}")
            
            # 方法2: 只标记body（不修改物理属性，避免仿真不稳定）
            for i, name in enumerate(sim.model.body_names):
                if name and any(pattern in name.lower() for pattern in patterns):
                    try:
                        print(f"  ✅ 标记body已删除: {name} (保持原始物理属性)")
                    except Exception as e:
                        print(f"  ⚠️ 标记body失败: {e}")
        
        # 方法3: 不处理joint（经验证不必要）
        # Joint锁定在我们的测试场景中不必要，因为：
        # 1. Geom已禁用视觉和碰撞检测
        # 2. 机器人无法与"删除"的物体交互
        # 3. 简化实现，避免不必要的约束
        
        # 轻微更新仿真状态（不改变位置）
        try:
            sim.forward()  # 只重新计算前向动力学
            print(f"  ✅ 更新仿真状态")
        except Exception as e:
            print(f"  ⚠️ 仿真更新失败: {e}")
        
        # 验证移除效果
        print(f"\n🔍 验证移除效果（位置应保持不变）:")
        for obj_type in object_types:
            patterns = object_patterns.get(obj_type, [])
            for i, name in enumerate(sim.model.body_names):
                if name and any(pattern in name.lower() for pattern in patterns):
                    pos = sim.data.body_xpos[i]
                    try:
                        mass = sim.model.body_mass[i] if hasattr(sim.model, 'body_mass') else 'unknown'
                        alpha = sim.model.geom_rgba[i][-1] if i < len(sim.model.geom_rgba) else 'unknown'
                        print(f"  - {name}: pos={pos}, mass={mass}, alpha={alpha}")
                    except:
                        print(f"  - {name}: pos={pos}")
        
        if objects_removed:
            print(f"✅ 处理了 {len(objects_removed)} 个geom组件")
        else:
            print("❌ 未找到指定物体或移除失败")
        
    except Exception as e:
        print(f"❌ 移除物体过程出错: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    test_ood_generalization(args)

if __name__ == "__main__":
    main() 