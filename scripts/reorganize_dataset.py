#!/usr/bin/env python3
"""
Reorganize Plant_Village_Blueberry dataset to standard structure.
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

def normalize_filename(filename):
    """Normalize filename by removing extension."""
    return Path(filename).stem

def json_to_csv(json_file, csv_file):
    """Convert JSON annotation to CSV format."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    if not annotations:
        # Create empty CSV file
        with open(csv_file, 'w') as f:
            f.write("#item,x,y,width,height,label\n")
        return
    
    with open(csv_file, 'w') as f:
        f.write("#item,x,y,width,height,label\n")
        for idx, ann in enumerate(annotations):
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                category_id = ann.get('category_id', 1)
                # Map category_id to label (1 for foreground, 0 for background)
                label = 1 if category_id != 0 else 0
                f.write(f"{idx},{x},{y},{w},{h},{label}\n")

def create_labelmap(category_name, output_file):
    """Create labelmap.json file."""
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
            "object_name": category_name
        }
    ]
    with open(output_file, 'w') as f:
        json.dump(labelmap, f, indent=2)

def reorganize_blueberries(root):
    """Reorganize Blueberry___healthy category."""
    print("Reorganizing blueberries...")
    
    source_dir = root / 'Blueberry___healthy' / 'without_augmentation'
    target_images = root / 'blueberries' / 'images'
    target_json = root / 'blueberries' / 'json'
    target_csv = root / 'blueberries' / 'csv'
    
    if not source_dir.exists():
        print(f"  Warning: {source_dir} does not exist")
        return
    
    # Get all JPG files
    jpg_files = list(source_dir.glob('*.JPG'))
    print(f"  Found {len(jpg_files)} JPG files")
    
    moved_count = 0
    for jpg_file in jpg_files:
        stem = jpg_file.stem
        json_file = source_dir / f"{stem}.json"
        
        # Copy image
        target_img = target_images / jpg_file.name
        if not target_img.exists():
            shutil.copy2(jpg_file, target_img)
        
        # Copy JSON if exists
        if json_file.exists():
            target_json_file = target_json / f"{stem}.json"
            if not target_json_file.exists():
                shutil.copy2(json_file, target_json_file)
            
            # Generate CSV
            csv_file = target_csv / f"{stem}.csv"
            if not csv_file.exists():
                json_to_csv(json_file, csv_file)
        
        moved_count += 1
        if moved_count % 500 == 0:
            print(f"  Processed {moved_count} files...")
    
    print(f"  Completed: {moved_count} files processed")
    
    # Create labelmap.json
    labelmap_file = root / 'blueberries' / 'labelmap.json'
    create_labelmap("blueberry", labelmap_file)
    print(f"  Created labelmap.json")

def reorganize_backgrounds(root):
    """Reorganize Background_without_leaves category."""
    print("Reorganizing backgrounds...")
    
    source_dir = root / 'Background_without_leaves' / 'without_augmentation'
    target_images = root / 'backgrounds' / 'images'
    target_csv = root / 'backgrounds' / 'csv'
    
    if not source_dir.exists():
        print(f"  Warning: {source_dir} does not exist")
        return
    
    # Get all jpg files
    jpg_files = list(source_dir.glob('*.jpg'))
    print(f"  Found {len(jpg_files)} jpg files")
    
    moved_count = 0
    for jpg_file in jpg_files:
        stem = jpg_file.stem
        
        # Copy image
        target_img = target_images / jpg_file.name
        if not target_img.exists():
            shutil.copy2(jpg_file, target_img)
        
        # Create empty CSV (background images have no annotations)
        csv_file = target_csv / f"{stem}.csv"
        if not csv_file.exists():
            with open(csv_file, 'w') as f:
                f.write("#item,x,y,width,height,label\n")
        
        moved_count += 1
        if moved_count % 500 == 0:
            print(f"  Processed {moved_count} files...")
    
    print(f"  Completed: {moved_count} files processed")
    
    # Create labelmap.json
    labelmap_file = root / 'backgrounds' / 'labelmap.json'
    create_labelmap("background", labelmap_file)
    print(f"  Created labelmap.json")

def build_filename_mapping(root):
    """Build mapping from original filenames to current filenames using JSON files."""
    print("Building filename mapping...")
    
    mapping = {}  # pvc_filename -> current_filename
    
    # Map from blueberries JSON files
    json_dir = root / 'blueberries' / 'json'
    if json_dir.exists():
        json_files = list(json_dir.glob('*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if data.get('images') and len(data['images']) > 0:
                    img_info = data['images'][0]
                    pvc_filename = img_info.get('pvc_filename', '')
                    current_filename = img_info.get('file_name', '')
                    if pvc_filename and current_filename:
                        # Remove extension for mapping
                        mapping[pvc_filename] = Path(current_filename).stem
            except Exception as e:
                pass
    
    print(f"  Built mapping for {len(mapping)} images")
    return mapping

def reorganize_sets(root):
    """Reorganize dataset split files."""
    print("Reorganizing dataset splits...")
    
    source_dir = root / 'all'
    if not source_dir.exists():
        print(f"  Warning: {source_dir} does not exist")
        return
    
    # Build filename mapping
    filename_mapping = build_filename_mapping(root)
    
    # Read split files
    for split_file in ['train.txt', 'val.txt', 'test.txt']:
        source_file = source_dir / split_file
        if not source_file.exists():
            continue
        
        # Read image names (these are original pvc_filenames)
        with open(source_file, 'r') as f:
            original_names = [line.strip() for line in f if line.strip()]
        
        # Determine which category each image belongs to
        blueberries_images = []
        backgrounds_images = []
        
        blueberries_dir = root / 'blueberries' / 'images'
        backgrounds_dir = root / 'backgrounds' / 'images'
        
        for original_name in original_names:
            # Remove extension for matching
            original_stem = Path(original_name).stem
            
            # Try to find using mapping first
            current_stem = filename_mapping.get(original_name, None)
            if current_stem:
                # Check if file exists in blueberries
                for ext in ['.JPG', '.jpg']:
                    if (blueberries_dir / f"{current_stem}{ext}").exists():
                        blueberries_images.append(current_stem)
                        break
                continue
            
            # Fallback: try direct matching
            found = False
            for ext in ['.JPG', '.jpg', '.png', '.PNG']:
                if (blueberries_dir / f"{original_stem}{ext}").exists():
                    blueberries_images.append(original_stem)
                    found = True
                    break
                elif (backgrounds_dir / f"{original_stem}{ext}").exists():
                    backgrounds_images.append(original_stem)
                    found = True
                    break
            
            if not found:
                # Try matching with any extension in the directory
                for img_file in list(blueberries_dir.glob('*')):
                    if img_file.stem == original_stem or img_file.name == original_name:
                        blueberries_images.append(img_file.stem)
                        found = True
                        break
                if not found:
                    for img_file in list(backgrounds_dir.glob('*')):
                        if img_file.stem == original_stem or img_file.name == original_name:
                            backgrounds_images.append(img_file.stem)
                            found = True
                            break
        
        # Write split files for each category
        if blueberries_images:
            target_file = root / 'blueberries' / 'sets' / split_file
            with open(target_file, 'w') as f:
                for img_name in blueberries_images:
                    f.write(f"{img_name}\n")
            print(f"  Created blueberries/sets/{split_file} with {len(blueberries_images)} images")
        
        if backgrounds_images:
            target_file = root / 'backgrounds' / 'sets' / split_file
            with open(target_file, 'w') as f:
                for img_name in backgrounds_images:
                    f.write(f"{img_name}\n")
            print(f"  Created backgrounds/sets/{split_file} with {len(backgrounds_images)} images")
    
    # Create all.txt files
    for category in ['blueberries', 'backgrounds']:
        images_dir = root / category / 'images'
        if images_dir.exists():
            all_images = []
            for ext in ['.JPG', '.jpg', '.png', '.PNG']:
                all_images.extend([f.stem for f in images_dir.glob(f'*{ext}')])
            
            if all_images:
                all_file = root / category / 'sets' / 'all.txt'
                with open(all_file, 'w') as f:
                    for img_name in sorted(set(all_images)):
                        f.write(f"{img_name}\n")
                print(f"  Created {category}/sets/all.txt with {len(set(all_images))} images")

def main():
    root = Path('/home/yuhanlin/Database/local/Plant_Village_Blueberry')
    
    print("=" * 60)
    print("Reorganizing Plant_Village_Blueberry dataset")
    print("=" * 60)
    
    reorganize_blueberries(root)
    print()
    reorganize_backgrounds(root)
    print()
    reorganize_sets(root)
    
    print()
    print("=" * 60)
    print("Reorganization complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()
