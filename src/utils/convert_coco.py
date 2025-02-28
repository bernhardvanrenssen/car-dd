import os
import shutil
import argparse
from pycocotools.coco import COCO

def convert_coco(json_file, images_base_dir, output_dir, split_name):
    """
    Converts a COCO JSON file into a folder structure:
    output_dir/split_name/<category_name>/image.jpg
    Assumes the raw images are stored under images_base_dir/split_name.
    """
    # Load COCO annotations
    coco = COCO(json_file)
    # Create a mapping from category id to category name
    categories = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    print("Categories found:", categories)

    # Create output subfolders for this split and for each category
    for cat_name in categories.values():
        cat_folder = os.path.join(output_dir, split_name, cat_name)
        os.makedirs(cat_folder, exist_ok=True)

    # Build a mapping from image id to file name
    img_ids = coco.getImgIds()
    imgs = coco.loadImgs(img_ids)
    imgid_to_filename = {img['id']: img['file_name'] for img in imgs}

    # Group annotations by image id
    ann_ids = coco.getAnnIds()
    anns = coco.loadAnns(ann_ids)
    img_annotations = {}
    for ann in anns:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Copy images into the appropriate category folder.
    # Here we select the first annotation's category as the label.
    for img_id, ann_list in img_annotations.items():
        chosen_ann = ann_list[0]
        cat_id = chosen_ann['category_id']
        cat_name = categories[cat_id]
        file_name = imgid_to_filename[img_id]
        # Since images are split into subdirectories, construct the source path accordingly.
        src_path = os.path.join(images_base_dir, split_name, file_name)
        dst_path = os.path.join(output_dir, split_name, cat_name, file_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {src_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO JSON annotations to a folder structure by category")
    parser.add_argument("--json_file", type=str, required=True, help="Path to the COCO JSON file")
    parser.add_argument("--images_base_dir", type=str, required=True, help="Base directory containing the split folders for images (e.g., data/COCO/images)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data (e.g., data/COCO/processed)")
    parser.add_argument("--split_name", type=str, required=True, help="Name of the split (train, val, or test)")
    args = parser.parse_args()
    
    convert_coco(args.json_file, args.images_base_dir, args.output_dir, args.split_name)
