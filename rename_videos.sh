#!/bin/bash

# 进入视频目录
cd ood_test_videos

echo "🔄 重命名视频文件，基于实际command内容..."

# === 最新成功的标准化测试 ===
# 1. 标准对照组
if [ -f "control_standard_task0_success.mp4" ]; then
    mv "control_standard_task0_success.mp4" "01_CONTROL_black_bowl_between_plate_ramekin_to_plate_SUCCESS.mp4"
    echo "✅ 重命名: 标准黑碗任务0"
fi

if [ -f "control_standard_task1_success.mp4" ]; then
    mv "control_standard_task1_success.mp4" "02_CONTROL_black_bowl_next_to_ramekin_to_plate_SUCCESS.mp4"
    echo "✅ 重命名: 标准黑碗任务1"
fi

# 2. 物体泛化OOD - 成功案例
if [ -f "ood_ramekin_between_success.mp4" ]; then
    mv "ood_ramekin_between_success.mp4" "03_OOD_OBJECT_ramekin_between_plate_black_bowl_to_plate_SUCCESS.mp4"
    echo "🔄 重命名: ramekin操作成功"
fi

if [ -f "ood_plate_next_to_success.mp4" ]; then
    mv "ood_plate_next_to_success.mp4" "04_OOD_OBJECT_plate_next_to_ramekin_to_black_bowl_SUCCESS.mp4"
    echo "🔄 重命名: plate操作成功"
fi

if [ -f "ood_ramekin_center_success.mp4" ]; then
    mv "ood_ramekin_center_success.mp4" "05_OOD_OBJECT_ramekin_from_table_center_to_plate_SUCCESS.mp4"
    echo "🔄 重命名: ramekin中心操作成功"
fi

# 3. 复杂OOD - 意外成功
if [ -f "ood_stove_non_standard_success.mp4" ]; then
    mv "ood_stove_non_standard_success.mp4" "06_OOD_COMPLEX_stove_to_plate_SUCCESS.mp4"
    echo "🔥 重命名: stove操作意外成功"
fi

if [ -f "ood_non_standard_target_success.mp4" ]; then
    mv "ood_non_standard_target_success.mp4" "07_OOD_COMPLEX_black_bowl_closest_to_wooden_cabinet_SUCCESS.mp4"
    echo "🏠 重命名: wooden cabinet目标成功"
fi

# 4. 位置OOD - 失败案例
if [ -f "ood_place_on_ramekin_failure.mp4" ]; then
    mv "ood_place_on_ramekin_failure.mp4" "08_OOD_PLACE_black_bowl_between_to_ramekin_FAILURE.mp4"
    echo "📍 重命名: 放到ramekin失败 (物理限制)"
fi

# === 早期非标准格式测试 (全部失败) ===
echo ""
echo "📁 重命名早期非标准格式测试文件..."

# 早期失败的非标准格式测试
if [ -f "ood_stove_manipulation_failure.mp4" ]; then
    mv "ood_stove_manipulation_failure.mp4" "EARLY_NON_STANDARD_stove_manipulation_FAILURE.mp4"
    echo "❌ 重命名: 早期stove测试失败"
fi

if [ -f "ood_plate_manipulation_failure.mp4" ]; then
    mv "ood_plate_manipulation_failure.mp4" "EARLY_NON_STANDARD_plate_to_wooden_cabinet_FAILURE.mp4"
    echo "❌ 重命名: 早期plate测试失败"
fi

if [ -f "ood_cookie_box_manipulation_failure.mp4" ]; then
    mv "ood_cookie_box_manipulation_failure.mp4" "EARLY_NON_STANDARD_cookie_box_to_stove_FAILURE.mp4"
    echo "❌ 重命名: 早期cookie box测试失败"
fi

if [ -f "control_stove_task_black_bowl_failure.mp4" ]; then
    mv "control_stove_task_black_bowl_failure.mp4" "EARLY_NON_STANDARD_black_bowl_closest_to_plate_FAILURE.mp4"
    echo "❌ 重命名: 早期黑碗测试失败"
fi

if [ -f "control_plate_task_black_bowl_failure.mp4" ]; then
    mv "control_plate_task_black_bowl_failure.mp4" "EARLY_NON_STANDARD_black_bowl_on_table_to_cabinet_FAILURE.mp4"
    echo "❌ 重命名: 早期桌上黑碗测试失败"
fi

if [ -f "control_cookie_task_black_bowl_failure.mp4" ]; then
    mv "control_cookie_task_black_bowl_failure.mp4" "EARLY_NON_STANDARD_black_bowl_visible_to_stove_FAILURE.mp4"
    echo "❌ 重命名: 早期场景黑碗测试失败"
fi

if [ -f "ood_place_bowl_on_bowl_failure.mp4" ]; then
    mv "ood_place_bowl_on_bowl_failure.mp4" "EARLY_NON_STANDARD_black_bowl_left_on_other_bowl_FAILURE.mp4"
    echo "❌ 重命名: 早期黑碗叠放测试失败"
fi

if [ -f "control_place_task_normal_failure.mp4" ]; then
    mv "control_place_task_normal_failure.mp4" "EARLY_NON_STANDARD_black_bowl_left_to_wooden_cabinet_FAILURE.mp4"
    echo "❌ 重命名: 早期黑碗到木柜测试失败"
fi

echo ""
echo "🎉 视频文件重命名完成！"
echo "📊 重命名结果："
echo "   ✅ 标准LIBERO格式测试: 成功率极高"
echo "   ❌ 非标准格式测试: 全部失败"
echo "   🔍 文件名现在包含实际的command内容"

ls -la *.mp4
