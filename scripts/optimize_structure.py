#!/usr/bin/env python3
"""
优化 Plant Village Blueberry 数据集结构
根据 acfr-multifruit-2016 数据集结构规范
将当前结构重组为子类别组织结构
"""

import os
import json
import shutil
from pathlib import Path

def create_subcategory_structure(root_dir):
    """创建子类别目录结构"""
    root = Path(root_dir)
    blueberries_dir = root / "blueberries"
    
    # 创建子类别目录
    subcategories = ["healthy", "background"]
    for subcat in subcategories:
        subcat_dir = blueberries_dir / subcat
        (subcat_dir / "csv").mkdir(parents=True, exist_ok=True)
        (subcat_dir / "json").mkdir(parents=True, exist_ok=True)
        (subcat_dir / "images").mkdir(parents=True, exist_ok=True)
    
    return blueberries_dir, subcategories

def normalize_filename(filename):
    """规范化文件名，移除扩展名并处理特殊字符"""
    stem = Path(filename).stem
    # 移除空格和特殊字符，保留字母数字和下划线
    normalized = stem.replace(" ", "_").replace("(", "").replace(")", "")
    return normalized

def move_to_subcategory(source_dir, target_subcat_dir, subcat_name):
    """将源目录的内容移动到子类别目录"""
    source = Path(source_dir)
    target = Path(target_subcat_dir)
    
    print(f"\n处理子类别: {subcat_name}")
    print(f"  源目录: {source}")
    print(f"  目标目录: {target}")
    
    # 移动图像文件
    if (source / "images").exists():
        image_count = 0
        for img_file in (source / "images").iterdir():
            if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                # 规范化文件名，统一使用.jpg扩展名
                stem = normalize_filename(img_file.name)
                new_name = f"{stem}.jpg"
                target_img = target / "images" / new_name
                
                try:
                    shutil.copy2(img_file, target_img)
                    image_count += 1
                except Exception as e:
                    print(f"  警告: 无法复制图像 {img_file}: {e}")
        
        print(f"  移动了 {image_count} 张图像")
    
    # 移动CSV文件
    if (source / "csv").exists():
        csv_count = 0
        for csv_file in (source / "csv").iterdir():
            if csv_file.is_file() and csv_file.suffix.lower() == '.csv':
                # 规范化文件名
                stem = normalize_filename(csv_file.name)
                new_name = f"{stem}.csv"
                target_csv = target / "csv" / new_name
                
                try:
                    shutil.copy2(csv_file, target_csv)
                    csv_count += 1
                except Exception as e:
                    print(f"  警告: 无法复制CSV {csv_file}: {e}")
        
        print(f"  移动了 {csv_count} 个CSV文件")
    
    # 移动JSON文件
    if (source / "json").exists():
        json_count = 0
        for json_file in (source / "json").iterdir():
            if json_file.is_file() and json_file.suffix.lower() == '.json':
                # 规范化文件名
                stem = normalize_filename(json_file.name)
                new_name = f"{stem}.json"
                target_json = target / "json" / new_name
                
                try:
                    shutil.copy2(json_file, target_json)
                    json_count += 1
                except Exception as e:
                    print(f"  警告: 无法复制JSON {json_file}: {e}")
        
        print(f"  移动了 {json_count} 个JSON文件")

def create_unified_labelmap(blueberries_dir):
    """创建统一的labelmap.json，包含所有子类别"""
    labelmap = [
        {
            "object_id": 0,
            "label_id": 0,
            "keyboard_shortcut": "0",
            "object_name": "background"
        },
        {
            "object_id": 1,
            "label_id": 1,
            "keyboard_shortcut": "1",
            "object_name": "healthy"
        },
        {
            "object_id": 2,
            "label_id": 2,
            "keyboard_shortcut": "2",
            "object_name": "background_without_leaves"
        }
    ]
    
    labelmap_path = blueberries_dir / "labelmap.json"
    with open(labelmap_path, 'w', encoding='utf-8') as f:
        json.dump(labelmap, f, indent=2, ensure_ascii=False)
    
    print(f"\n创建统一的 labelmap.json: {labelmap_path}")

def merge_splits(root_dir, blueberries_dir):
    """合并数据集划分文件到主类别目录"""
    # 注意：由于我们已经移动了文件，需要从新的子类别目录中收集所有图像
    target_sets = blueberries_dir / "sets"
    target_sets.mkdir(parents=True, exist_ok=True)
    
    print(f"\n合并数据集划分文件...")
    
    # 收集所有图像文件名（从两个子类别目录）
    healthy_images = set()
    background_images = set()
    
    healthy_dir = blueberries_dir / "healthy" / "images"
    background_dir = blueberries_dir / "background" / "images"
    
    if healthy_dir.exists():
        for img_file in healthy_dir.iterdir():
            if img_file.is_file():
                stem = normalize_filename(img_file.name)
                healthy_images.add(stem)
    
    if background_dir.exists():
        for img_file in background_dir.iterdir():
            if img_file.is_file():
                stem = normalize_filename(img_file.name)
                background_images.add(stem)
    
    all_images = healthy_images | background_images
    print(f"  找到 {len(healthy_images)} 张healthy图像, {len(background_images)} 张background图像")
    print(f"  总计 {len(all_images)} 张图像")
    
    # 读取原始的划分文件（如果存在）
    blueberries_sets = Path(root_dir) / "blueberries" / "sets"
    backgrounds_sets = Path(root_dir) / "backgrounds" / "sets"
    
    # 处理每个划分文件
    for split_name in ['train', 'val', 'test']:
        blueberries_split = blueberries_sets / f"{split_name}.txt"
        backgrounds_split = backgrounds_sets / f"{split_name}.txt"
        target_split = target_sets / f"{split_name}.txt"
        
        split_stems = set()
        
        # 读取blueberries的划分
        if blueberries_split.exists():
            with open(blueberries_split, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    stem = normalize_filename(Path(line).stem)
                    if stem in healthy_images:
                        split_stems.add(stem)
        
        # 读取backgrounds的划分（如果有）
        if backgrounds_split.exists():
            with open(backgrounds_split, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    stem = normalize_filename(Path(line).stem)
                    if stem in background_images:
                        split_stems.add(stem)
        
        # 如果没有划分文件，根据比例分配（仅作为后备方案）
        if not split_stems and split_name in ['train', 'val', 'test']:
            # 这里不自动分配，保持为空，让用户手动创建
            pass
        
        # 写入划分文件
        if split_stems:
            with open(target_split, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sorted(split_stems)) + '\n')
            print(f"  创建 {split_name}.txt ({len(split_stems)} 个文件)")
    
    # 创建 all.txt（包含所有图像）
    all_file = target_sets / "all.txt"
    with open(all_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(all_images)) + '\n')
    print(f"  创建 all.txt ({len(all_images)} 个文件)")
    
    # 创建 train_val.txt（train + val）
    train_val_file = target_sets / "train_val.txt"
    train_file = target_sets / "train.txt"
    val_file = target_sets / "val.txt"
    if train_file.exists() and val_file.exists():
        train_stems = set()
        val_stems = set()
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_stems = {line.strip() for line in f if line.strip()}
        with open(val_file, 'r', encoding='utf-8') as f:
            val_stems = {line.strip() for line in f if line.strip()}
        
        train_val_stems = train_stems | val_stems
        with open(train_val_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(train_val_stems)) + '\n')
        print(f"  创建 train_val.txt ({len(train_val_stems)} 个文件)")

def main():
    root_dir = Path(__file__).parent.parent
    
    print("=" * 60)
    print("优化 Plant Village Blueberry 数据集结构")
    print("=" * 60)
    
    # 创建子类别目录结构
    blueberries_dir, subcategories = create_subcategory_structure(root_dir)
    print(f"\n创建子类别目录结构: {blueberries_dir}")
    
    # 将blueberries目录的内容移动到healthy子类别
    blueberries_source = root_dir / "blueberries"
    healthy_target = blueberries_dir / "healthy"
    if blueberries_source.exists():
        print(f"\n将 blueberries 目录重组为 healthy 子类别...")
        # 移动现有内容到healthy子类别
        move_to_subcategory(blueberries_source, healthy_target, "healthy")
    
    # 将backgrounds目录的内容移动到background子类别
    backgrounds_source = root_dir / "backgrounds"
    background_target = blueberries_dir / "background"
    if backgrounds_source.exists():
        print(f"\n将 backgrounds 目录重组为 background 子类别...")
        move_to_subcategory(backgrounds_source, background_target, "background")
    
    # 创建统一的labelmap.json
    create_unified_labelmap(blueberries_dir)
    
    # 合并数据集划分文件
    merge_splits(root_dir, blueberries_dir)
    
    print("\n" + "=" * 60)
    print("数据集结构优化完成！")
    print("=" * 60)
    print(f"\n新结构位于: {blueberries_dir}")
    print("\n新的目录结构:")
    print("  blueberries/")
    print("    ├── healthy/")
    print("    │   ├── csv/")
    print("    │   ├── json/")
    print("    │   └── images/")
    print("    ├── background/")
    print("    │   ├── csv/")
    print("    │   ├── json/")
    print("    │   └── images/")
    print("    ├── labelmap.json")
    print("    └── sets/")
    print("\n下一步:")
    print("1. 检查优化后的数据")
    print("2. 更新 convert_to_coco.py 脚本以支持新结构")
    print("3. 重新生成 COCO 格式标注")
    print("4. 更新 README.md 文档")

if __name__ == "__main__":
    main()

