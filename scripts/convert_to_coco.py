#!/usr/bin/env python3
"""
Convert Plant Village Blueberry dataset annotations to COCO JSON format.
Supports multi-class classification (2 classes: healthy, background).
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image

def read_split_list(split_file: Path) -> List[str]:
    """Read image base names (without extension) from a split file."""
    if not split_file.exists():
        return []
    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]

def image_size(image_path: Path) -> Tuple[int, int]:
    """Return (width, height) for an image path using PIL."""
    with Image.open(image_path) as img:
        return img.width, img.height

def parse_csv_boxes(csv_path: Path) -> List[Dict]:
    """Parse a single CSV file and return bounding boxes with category IDs."""
    if not csv_path.exists():
        return []
    
    boxes = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Skip comment lines
                if row.get('#item') is None and 'item' not in row:
                    continue
                
                # Get item index (may be in '#item' or 'item' column)
                item_key = '#item' if '#item' in row else 'item'
                if item_key not in row:
                    continue
                
                x = float(row.get('x', 0))
                y = float(row.get('y', 0))
                width = float(row.get('width', row.get('w', row.get('dx', 0))))
                height = float(row.get('height', row.get('h', row.get('dy', 0))))
                label = int(row.get('label', 1))
                
                # For classification tasks, if bbox is [0,0,width,height], it's a full-image annotation
                if width > 0 and height > 0:
                    boxes.append({
                        'bbox': [x, y, width, height],
                        'area': width * height,
                        'category_id': label
                    })
            except (ValueError, KeyError) as e:
                continue
    
    return boxes

def collect_annotations_for_split(
    category_root: Path,
    split: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Collect COCO dictionaries for images, annotations, and categories.
    Supports structure: blueberries/{subcategory}/ subdirectories.
    """
    sets_dir = category_root / "sets"
    split_file = sets_dir / f"{split}.txt"
    image_stems = set(read_split_list(split_file))
    
    if not image_stems:
        # Fall back to all images if no split file
        image_stems = set()
        for subcat_dir in category_root.glob("*"):
            if subcat_dir.is_dir() and subcat_dir.name != "sets":
                images_dir = subcat_dir / "images"
                if images_dir.exists():
                    image_stems.update({p.stem for p in images_dir.glob("*.png")})
                    image_stems.update({p.stem for p in images_dir.glob("*.jpg")})
                    image_stems.update({p.stem for p in images_dir.glob("*.JPG")})
    
    images: List[Dict] = []
    anns: List[Dict] = []
    categories: List[Dict] = [
        {"id": 1, "name": "healthy", "supercategory": "blueberry"},
        {"id": 2, "name": "background_without_leaves", "supercategory": "blueberry"}
    ]
    
    image_id_counter = 1
    ann_id_counter = 1
    
    # Check all subcategory directories
    subcategory_names = ['healthy', 'background']
    
    for stem in sorted(image_stems):
        # Try each subcategory directory
        img_path = None
        subcategory = None
        csv_path = None
        
        for subcat_name in subcategory_names:
            subcat_dir = category_root / subcat_name
            for ext in ['.png', '.jpg', '.JPG', '.PNG', '.jpeg', '.JPEG']:
                test_path = subcat_dir / 'images' / f"{stem}{ext}"
                if test_path.exists():
                    img_path = test_path
                    subcategory = subcat_name
                    csv_path = subcat_dir / 'csv' / f"{stem}.csv"
                    break
            if img_path:
                break
        
        if not img_path:
            continue
        
        width, height = image_size(img_path)
        images.append({
            "id": image_id_counter,
            "file_name": f"blueberries/{subcategory}/images/{img_path.name}",
            "width": width,
            "height": height,
        })
        
        # Determine category_id based on subcategory
        category_id_map = {
            'healthy': 1,
            'background': 2
        }
        default_category_id = category_id_map.get(subcategory, 1)
        
        if csv_path and csv_path.exists():
            boxes = parse_csv_boxes(csv_path)
            if boxes:
                for box in boxes:
                    # Use category_id from CSV if available, otherwise use subcategory-based ID
                    cat_id = box.get('category_id', default_category_id)
                    anns.append({
                        "id": ann_id_counter,
                        "image_id": image_id_counter,
                        "category_id": cat_id,
                        "bbox": box['bbox'],
                        "area": box['area'],
                        "iscrowd": 0,
                    })
                    ann_id_counter += 1
            else:
                # No boxes found, create full-image annotation for classification
                anns.append({
                    "id": ann_id_counter,
                    "image_id": image_id_counter,
                    "category_id": default_category_id,
                    "bbox": [0, 0, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                })
                ann_id_counter += 1
        else:
            # No CSV file, create full-image annotation
            anns.append({
                "id": ann_id_counter,
                "image_id": image_id_counter,
                "category_id": default_category_id,
                "bbox": [0, 0, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            ann_id_counter += 1
        
        image_id_counter += 1
    
    return images, anns, categories

def build_coco_dict(
    images: List[Dict],
    anns: List[Dict],
    categories: List[Dict],
    description: str,
) -> Dict:
    """Build a complete COCO dict from components."""
    return {
        "info": {
            "year": 2025,
            "version": "1.0.0",
            "description": description,
            "url": "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset",
        },
        "licenses": [
            {
                "id": 1,
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/"
            }
        ],
        "images": images,
        "annotations": anns,
        "categories": categories,
    }

def main():
    parser = argparse.ArgumentParser(
        description="Convert Plant Village Blueberry CSV annotations to COCO JSON format"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root directory of the dataset (default: current directory)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="annotations",
        help="Output directory for COCO JSON files (default: annotations)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process (default: train val test)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also generate combined COCO JSON files across all subcategories",
    )
    
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    category_root = root / "blueberries"
    if not category_root.exists():
        print(f"Error: Category directory not found: {category_root}")
        sys.exit(1)
    
    print(f"Converting Plant Village Blueberry dataset to COCO format...")
    print(f"Root: {root}")
    print(f"Output: {out_dir}")
    print(f"Splits: {args.splits}")
    
    for split in args.splits:
        print(f"\nProcessing {split} split...")
        images, anns, categories = collect_annotations_for_split(category_root, split)
        
        if not images:
            print(f"  Warning: No images found for {split} split")
            continue
        
        description = f"Plant Village Blueberry {split} split"
        coco_dict = build_coco_dict(images, anns, categories, description)
        
        output_file = out_dir / f"blueberries_instances_{split}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2, ensure_ascii=False)
        
        print(f"  Created: {output_file}")
        print(f"  Images: {len(images)}")
        print(f"  Annotations: {len(anns)}")
        print(f"  Categories: {len(categories)}")
    
    if args.combined:
        print(f"\nGenerating combined COCO files...")
        # For combined files, we can merge all splits or create per-split combined files
        # Here we create per-split combined files (same as individual since we only have one category)
        for split in args.splits:
            images, anns, categories = collect_annotations_for_split(category_root, split)
            if images:
                description = f"Plant Village Blueberry {split} split (combined)"
                coco_dict = build_coco_dict(images, anns, categories, description)
                output_file = out_dir / f"combined_instances_{split}.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(coco_dict, f, indent=2, ensure_ascii=False)
                print(f"  Created: {output_file}")
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()
