import json
import os
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to Qwen1.5 format for visual_prompting.")
    parser.add_argument("input", help="Path to COCO annotation JSON file.")
    parser.add_argument("output", help="Output JSON file path.")
    parser.add_argument("--root", default="", help="Directory containing images referenced in the COCO annotations.")
    return parser.parse_args()

# === Helpers ===
def normalize_bbox_to_poly(bbox, w, h):
    x, y, bw, bh = bbox
    return [
        x / w, y / h,
        (x + bw) / w, y / h,
        (x + bw) / w, (y + bh) / h,
        x / w, (y + bh) / h
    ]

def normalize_segmentation(seg, w, h):
    return [
        [pt / w if i % 2 == 0 else pt / h for i, pt in enumerate(poly)]
        for poly in seg if isinstance(poly, list)
    ]
    
def main():
    args = parse_args()
    # ==== Load COCO ====
    with open(args.input, "r") as f:
        coco_data = json.load(f)

    # ID → image info
    id_to_image = {
        img["id"]: {
            "image": img["file_name"],
            "height": img["height"],
            "width": img["width"]
        } for img in coco_data["images"]
    }


    # === Group annotations by image_id ===
    anns_by_image = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # === Convert ===
    qwen_entries = []

    for image_id, anns in anns_by_image.items():
        img_info = images_info[image_id]
        filename = img_info["file_name"]
        height = img_info["height"]
        width = img_info["width"]

        bboxes = []
        segms = []
        conversations = []

        for i, ann in enumerate(anns):            
            # Pick first sentence if exists
            describe = data['annotations'][0]['sentences'][0]['raw']
            gpt_answer = f"The object appears in the box to be {category_name}."

            # Normalized bbox polygon
            bbox_poly = normalize_bbox_to_poly(ann["bbox"], width, height)
            bboxes.append(bbox_poly)

            # Normalized segmentation
            seg_poly = normalize_segmentation(ann.get("segmentation", []), width, height)
            segms.append(seg_poly)

            # Add conversations
            conversations.append({
                "from": "human",
                "value": "<image>\nDescribe the object in <annotation>."
            })
            conversations.append({
                "from": "gpt",
                "value": gpt_answer
            })

        entry = {
            "id": image_id,
            "image": os.path.join(image_dir, filename),
            "image_size": [height, width],
            "bbox": bboxes,
            "segmentation": segms,
            "conversations": conversations
        }
        qwen_entries.append(entry)

    # === Save ===
    with open(output_path, "w") as f:
        json.dump(qwen_entries, f, indent=2)

if __name__ == "__main__":
    main()
