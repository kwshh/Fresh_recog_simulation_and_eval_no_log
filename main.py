"""
纯图片仿真测试（无日志版）
==========================
只需要两个目录：
  - 学习集目录：每个子文件夹 = 一个品类（格式: 条码_名称 或 纯名称）
  - 测试集目录：每个子文件夹 = 一个品类，图片按时间戳命名

流程：
  Phase 1: 从学习集建库（固定特征池）
  Phase 2: 按图片文件名时间戳排序，逐张推理 + 动态更新
  Phase 3: 生成可视化分析
"""

# ============================================================
# 导入依赖
# ============================================================

import os
import json
import csv
import re
from collections import defaultdict, Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# ============================================================
# 【配置区】
# ============================================================

# --- 模型配置 ---
ONNX_PATH = r"D:\deploy\rknn_model_zoo-main\weights\fresh_algorithm_iteration_version\0424_v4\best_model_224_embed.onnx"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 1280

# --- 分类映射关系表 ---
CLASS_MATCH_CSV = r"D:\dataset\LocalTest\Snacks\第二阶段_数据映射表 - Snack_Test.csv"
SHOP_NAME = "临沂家和超市"

# --- 数据路径 ---
STUDY_IMAGE_DIR = r"D:\dataset\LocalTest\Snacks\jiahe_shop_data\learn_split"
TEST_IMAGE_DIR = r"D:\dataset\LocalTest\Snacks\jiahe_shop_data\test_split"

# --- 输出路径 ---
SAVE_DIR = r"D:\dataset\LocalTest\Snacks\jiahe_shop_data\eval_report_no_log"
REPORT_CSV = os.path.join(SAVE_DIR, "simulation_report_details.csv")
CLASS_REPORT_CSV = os.path.join(SAVE_DIR, "simulation_class_report.csv")

# --- FeatureMatcher 配置 ---
FIXED_CAPACITY = 10
ROLLING_CAPACITY = 15
POLICY = "fifo"
AGGREGATION_MODE = "hybrid"
AGGREGATION_MARGIN = 0.1
STUDY_NUM = 5


# ============================================================
# 导入本地模块
# ============================================================
from lib.OnnxImageInfer import ONNX_Image_Infer
from lib.FeatureMatcher_CPP import FeatureMatcher
 
 
# ============================================================
# Phase 0: 初始化
# ============================================================
print("🔧 初始化 ONNX 推理引擎...")
onnx_engine = ONNX_Image_Infer(
    onnx_model_path=ONNX_PATH,
    infer_device="GPU",
    color_mode="RGB",
    mean=MEAN,
    std=STD,
    is_category=False
)
print(f"  ✅ 模型: {os.path.basename(ONNX_PATH)}")
print(f"  特征维度: {FEATURE_DIM}")
 
 
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
 
 
def extract_feature_from_image(img_path, feat_dim=1280):
    register_image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    feat = onnx_engine.infer(register_image)
    return feat
 
 
def class_name_from_folder(folder_name):
    """从目录名解析类别名：'2604270000134_秋葵' -> '秋葵'"""
    folder_name = str(folder_name).strip()
    return folder_name.split('_', 1)[1].strip() if '_' in folder_name else folder_name
 
 
def barcode_from_folder(folder_name):
    """从目录名提取条码：'2604270000134_秋葵' -> '2604270000134'"""
    folder_name = str(folder_name).strip()
    if '_' in folder_name:
        bc = folder_name.split('_', 1)[0]
        return bc if bc.isdigit() else ''
    return ''
 
 
# ============================================================
# 品类映射加载
# ============================================================
def load_class_mapping(csv_path, shop_col_name):
    """
    从品类映射 CSV 加载门店细类名 -> (类目, 粗类, 合并细类) 的映射。
    
    CSV 表头示例:
      类目, 二级类目, 粗类, 合并细类, ..., 发到家超市, ...
    
    Args:
        csv_path: CSV 文件路径
        shop_col_name: 门店列名（如 "发到家超市"）
    
    Returns:
        dict: {门店细类名: {'category': 类目, 'coarse': 粗类, 'fine': 合并细类}}
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"  ⚠️ 品类映射文件不存在: {csv_path}")
        return {}
 
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)
 
    # 找到门店列索引
    shop_col_idx = None
    for i, h in enumerate(headers):
        if h.strip() == shop_col_name:
            shop_col_idx = i
            break
 
    if shop_col_idx is None:
        valid_cols = [h.strip() for h in headers if h.strip()]
        print(f"  ⚠️ 未找到列 '{shop_col_name}'，可用列: {valid_cols}")
        return {}
 
    # 构建映射（类目/粗类空值继承上一行）
    mapping = {}
    last_category = ''
    last_coarse = ''
 
    for row in rows:
        if len(row) <= shop_col_idx:
            continue
 
        category = row[0].strip()
        coarse = row[2].strip() if len(row) > 2 else ''
        fine = row[3].strip() if len(row) > 3 else ''
        shop_val = row[shop_col_idx].strip()
 
        if category:
            last_category = category
        if coarse:
            last_coarse = coarse
 
        if shop_val:
            mapping[shop_val] = {
                'category': last_category,
                'coarse': last_coarse,
                'fine': fine,
            }
 
    print(f"  📋 加载品类映射: {len(mapping)} 条 (门店: {shop_col_name})")
 
    cats = Counter(v['category'] for v in mapping.values())
    for cat, cnt in cats.most_common():
        print(f"     {cat}: {cnt}")
 
    return mapping
 
 
# ============================================================
# Phase 1: 建库
# ============================================================
def phase1_initialize_fixed_pool(matcher, study_dir, feat_dim, study_num=5):
    print(f"\n>>> [Phase 1: 初始建库] 从 {study_dir} 加载固定特征...")
    if not os.path.exists(study_dir):
        print(f"  ❌ 目录不存在: {study_dir}")
        return 0, 0
 
    study_count = 0
    class_count = 0
    for folder_name in sorted(os.listdir(study_dir)):
        folder_path = os.path.join(study_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
 
        class_name = class_name_from_folder(folder_name)
        images = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(IMAGE_EXTS)])
 
        loaded = 0
        for img_name in images[:study_num]:
            img_path = os.path.join(folder_path, img_name)
            feat = extract_feature_from_image(img_path, feat_dim)
            matcher.update(feat, class_name, is_fixed=True)
            study_count += 1
            loaded += 1
 
        if loaded > 0:
            class_count += 1
 
    print(f"  ✅ 建库完成！{class_count} 个类别, {study_count} 张基准特征。")
    return class_count, study_count
 
 
# ============================================================
# Phase 2: 纯图片测试（无日志，带品类映射）
# ============================================================
def phase2_run_test(matcher, test_img_dir, feat_dim, study_dir,
                    output_report_path, output_class_report_path,
                    class_mapping=None):
    """
    按图片文件名时间戳排序，逐张推理 + 动态更新。
    不依赖任何日志文件，通过 class_mapping 附加品类信息。
    """
    print(f"\n>>> [Phase 2: 仿真测试] 扫描测试图片...")
 
    if class_mapping is None:
        class_mapping = {}
 
    # ---- 构建 名称->条码 映射表 ----
    name_to_barcode = {}
    if study_dir and os.path.exists(study_dir):
        for folder_name in os.listdir(study_dir):
            bc = barcode_from_folder(folder_name)
            nm = class_name_from_folder(folder_name)
            if bc and nm:
                name_to_barcode[nm] = bc
    for folder_name in os.listdir(test_img_dir):
        bc = barcode_from_folder(folder_name)
        nm = class_name_from_folder(folder_name)
        if bc and nm:
            name_to_barcode[nm] = bc
 
    def add_barcode(name):
        bc = name_to_barcode.get(name, '')
        return f"{bc}_{name}" if bc else name
 
    def get_class_info(name):
        """查询品类映射: 门店细类名 -> (类目, 粗类, 合并细类)"""
        info = class_mapping.get(name, None)
        if info:
            return info['category'], info['coarse'], info['fine']
        return '', '', ''
 
    print(f"  📦 构建了 {len(name_to_barcode)} 个 名称→条码 映射")
 
    # ---- 收集所有测试图片，按文件名排序 ----
    test_items = []
    for folder_name in sorted(os.listdir(test_img_dir)):
        folder_path = os.path.join(test_img_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
 
        ground_truth = class_name_from_folder(folder_name)
        for file_name in os.listdir(folder_path):
            if not file_name.lower().endswith(IMAGE_EXTS):
                continue
            img_path = os.path.join(folder_path, file_name)
            test_items.append((img_path, ground_truth, folder_name, file_name))
 
    test_items.sort(key=lambda x: x[3])
    print(f"  🖼️  共 {len(test_items)} 张测试图片，按文件名时序排列。")
    print(f"  开始仿真...")
 
    # ---- 统计变量 ----
    total_valid = 0
    top1_hits = 0
    top5_hits = 0
    total_output_count = 0
    report_data = []
    class_stats = {}
 
    # ---- 按品类映射统计（类目级别）----
    category_stats = defaultdict(lambda: {'total': 0, 'top1': 0, 'top5': 0})
    coarse_stats = defaultdict(lambda: {'total': 0, 'top1': 0, 'top5': 0})
 
    unmapped_names = set()
 
    # ---- 逐张测试 ----
    for i, (img_path, ground_truth, folder_name, file_name) in enumerate(test_items):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(test_items)}")
 
        feat = extract_feature_from_image(img_path, feat_dim)
 
        # 识别
        keep_results, margin12 = matcher.get_match_results(feat)
        pred_top_names = [cls for cls, score in keep_results]
        pred_top_scores = [score for cls, score in keep_results]
 
        pred_top5_names = pred_top_names[:5]
        pred_top5_scores = pred_top_scores[:5]
 
        total_valid += 1
        total_output_count += len(pred_top5_names)
 
        python_top1 = pred_top5_names[0] if pred_top5_names else "None"
        python_top1_score = pred_top5_scores[0] if pred_top5_scores else 0.0
 
        # 品类映射
        gt_category, gt_coarse, gt_fine = get_class_info(ground_truth)
        pred_category, pred_coarse, pred_fine = get_class_info(python_top1) if python_top1 != "None" else ('', '', '')
 
        if not gt_category and ground_truth not in unmapped_names:
            unmapped_names.add(ground_truth)
 
        # ---- 命中统计 ----
        is_top1_hit = False
        is_top5_hit = False
        is_coarse_hit = False  # 粗类命中（容错）
 
        if ground_truth not in class_stats:
            class_stats[ground_truth] = {'total': 0, 'top1': 0, 'top5': 0, 'coarse_hit': 0, 'output_count': 0}
 
        class_stats[ground_truth]['total'] += 1
        class_stats[ground_truth]['output_count'] += len(pred_top5_names)
 
        if len(pred_top5_names) > 0:
            if python_top1 == ground_truth:
                top1_hits += 1
                is_top1_hit = True
                class_stats[ground_truth]['top1'] += 1
 
            if ground_truth in pred_top5_names:
                top5_hits += 1
                is_top5_hit = True
                class_stats[ground_truth]['top5'] += 1
 
            # 粗类命中：预测的粗类 == 真实的粗类
            if gt_coarse and pred_coarse and gt_coarse == pred_coarse:
                is_coarse_hit = True
                class_stats[ground_truth]['coarse_hit'] += 1
 
        # 类目级别统计
        if gt_category:
            category_stats[gt_category]['total'] += 1
            if is_top1_hit:
                category_stats[gt_category]['top1'] += 1
            if is_top5_hit:
                category_stats[gt_category]['top5'] += 1
 
        if gt_coarse:
            coarse_stats[gt_coarse]['total'] += 1
            if is_top1_hit:
                coarse_stats[gt_coarse]['top1'] += 1
            if is_top5_hit:
                coarse_stats[gt_coarse]['top5'] += 1
 
        # ---- 输出 ----
        python_top1_with_bc = add_barcode(python_top1) if python_top1 != "None" else "None"
        python_top5_with_bc = [add_barcode(n) for n in pred_top5_names]
        ground_truth_with_bc = add_barcode(ground_truth)
 
        python_top5_display = []
        for n, s in zip(python_top5_with_bc, pred_top5_scores):
            python_top5_display.append(f"{n}({s:.4f})")
 
        image_rel = os.path.relpath(img_path, test_img_dir).replace(os.sep, '/')
 
        report_data.append({
            "image": image_rel,
            "ground_truth": ground_truth_with_bc,
            "gt_category": gt_category,
            "gt_coarse": gt_coarse,
            "gt_fine": gt_fine,
            "python_top1": python_top1_with_bc,
            "python_top1_score": round(python_top1_score, 4),
            "pred_category": pred_category,
            "pred_coarse": pred_coarse,
            "python_top5": str(python_top5_display),
            "python_output_count": len(pred_top5_names),
            "is_top1_hit": is_top1_hit,
            "is_top5_hit": is_top5_hit,
            "is_coarse_hit": is_coarse_hit,
            "margin12": round(margin12, 4) if margin12 else 0.0,
        })
 
        # 模拟点选
        matcher.update(feat, ground_truth, is_fixed=False)
 
    # ---- 导出明细 ----
    if report_data:
        with open(output_report_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=report_data[0].keys())
            writer.writeheader()
            writer.writerows(report_data)
        print(f"  📄 明细报告: {output_report_path}")
 
    # ---- 导出品类统计 ----
    class_report_data = []
    for cls_name, stats in class_stats.items():
        total = stats['total']
        cat, coarse, fine = get_class_info(cls_name)
        class_report_data.append({
            "品类名称": cls_name,
            "类目": cat,
            "粗类": coarse,
            "合并细类": fine,
            "测试样本": total,
            "Top1命中": stats['top1'],
            "Top5命中": stats['top5'],
            "粗类命中": stats['coarse_hit'],
            "Top1准确率": round(stats['top1'] / total, 4) if total else 0,
            "Top5准确率": round(stats['top5'] / total, 4) if total else 0,
            "粗类命中率": round(stats['coarse_hit'] / total, 4) if total else 0,
            "平均输出数": round(stats['output_count'] / total, 2) if total else 0,
        })
    class_report_data.sort(key=lambda x: x["测试样本"], reverse=True)
 
    if class_report_data:
        with open(output_class_report_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=class_report_data[0].keys())
            writer.writeheader()
            writer.writerows(class_report_data)
        print(f"  📈 品类报告: {output_class_report_path}")
 
    # ---- 打印总结 ----
    coarse_total_hits = sum(s['coarse_hit'] for s in class_stats.values())
 
    print(f"\n{'='*60}")
    print(f"  📊 仿真测试结果 (共 {total_valid} 有效样本, {len(class_stats)} 品类)")
    print(f"{'='*60}")
    if total_valid > 0:
        print(f"  Top-1 识别率:    {top1_hits/total_valid:.2%} ({top1_hits}/{total_valid})")
        print(f"  Top-5 召回率:    {top5_hits/total_valid:.2%} ({top5_hits}/{total_valid})")
        print(f"  粗类命中率:      {coarse_total_hits/total_valid:.2%} ({coarse_total_hits}/{total_valid})")
        print(f"  平均输出个数:    {total_output_count/total_valid:.2f}")
 
    # ---- 按类目统计 ----
    if category_stats:
        print(f"\n{'='*60}")
        print(f"  📂 按类目统计")
        print(f"{'='*60}")
        print(f"  {'类目':<10} {'样本':>6} {'Top1':>8} {'Top5':>8}")
        print(f"  {'-'*36}")
        for cat in sorted(category_stats.keys()):
            s = category_stats[cat]
            t = s['total']
            print(f"  {cat:<10} {t:>6} {s['top1']/t:>8.1%} {s['top5']/t:>8.1%}")
 
    # ---- 按粗类统计（最差的）----
    if coarse_stats:
        worst_coarse = [(name, s) for name, s in coarse_stats.items() if s['total'] >= 2]
        worst_coarse.sort(key=lambda x: x[1]['top1'] / x[1]['total'])
        worst_coarse = worst_coarse[:15]
 
        if worst_coarse:
            print(f"\n{'='*60}")
            print(f"  ⚠️  粗类Top1最差 Top{len(worst_coarse)}")
            print(f"{'='*60}")
            print(f"  {'粗类':<14} {'样本':>6} {'Top1':>8} {'Top5':>8}")
            print(f"  {'-'*40}")
            for name, s in worst_coarse:
                t = s['total']
                print(f"  {name:<14} {t:>6} {s['top1']/t:>8.1%} {s['top5']/t:>8.1%}")
 
    # ---- 未映射品类 ----
    if unmapped_names:
        print(f"\n  ⚠️ 未匹配品类映射: {len(unmapped_names)} 个")
        for name in sorted(unmapped_names):
            print(f"     {name}")
 
    print(f"{'='*60}")
 
    return report_data


# ============================================================
# 🚀 执行全流程
# ============================================================
 
os.makedirs(SAVE_DIR, exist_ok=True)
 
# ---- 加载品类映射 ----
print("\n>>> [Phase 0: 品类映射]")
class_mapping = load_class_mapping(CLASS_MATCH_CSV, SHOP_NAME)
 
# ---- Phase 1: 建库 ----
my_matcher = FeatureMatcher(
    fixed_capacity=FIXED_CAPACITY,
    rolling_capacity=ROLLING_CAPACITY,
    policy=POLICY,
    aggregation_mode=AGGREGATION_MODE,
    aggregation_margin=AGGREGATION_MARGIN
)
phase1_initialize_fixed_pool(my_matcher, STUDY_IMAGE_DIR, FEATURE_DIM, study_num=STUDY_NUM)
 
# ---- Phase 2: 测试 ----
report_data = phase2_run_test(
    my_matcher, TEST_IMAGE_DIR, FEATURE_DIM, STUDY_IMAGE_DIR,
    output_report_path=REPORT_CSV,
    output_class_report_path=CLASS_REPORT_CSV,
    class_mapping=class_mapping,
)
 
print(f"\n🎉 全部完成！结果在: {SAVE_DIR}")