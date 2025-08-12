import json, os
import numpy 
import argparse
from collections import defaultdict
from urllib.parse import urlparse
import cv2
from pycocotools import mask as mask_utils

#import pdb; pdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to Qwen1.5 format for visual_prompting.")
    parser.add_argument("input", help="Path to COCO annotation JSON file.")
    parser.add_argument("output", help="Output JSON file path.")
    parser.add_argument("--root", default="", help="Directory containing images referenced in the COCO annotations.")
    return parser.parse_args()


def normalize_bbox_to_poly(bbox, w, h):
    x, y, bw, bh = bbox
    return [x / w, y / h, bw / w, bh / h]


def normalize_segmentation(seg, w, h):

    return [
        [pt / w if i % 2 == 0 else pt / h for i, pt in enumerate(poly)]
        for poly in seg if isinstance(poly, list)
    ]

def get_polygons_from_RLE(rle):
    binary_mask = mask_utils.decode(rle)
    mask_uint8 = binary_mask.astype(numpy.uint8)

    # Find contours (external only)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    simplified = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)  # try 0.01–0.05 smaller with more details
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        simplified.append(approx)

    # Convert contours to polygon format (list of point lists)
    polygons = [cnt.reshape(-1, 2).tolist() for cnt in simplified]
    if len(polygons) > 0:
        return [numpy.array(poly + [poly[0]]).flatten().tolist() for poly in polygons]
    else:
        return []

def main():
    args = parse_args()

    # ==== Load COCO ====
    with open(args.input, "r") as f:
        coco_data = json.load(f)

    # Category ID → name (once)
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # ID → image info
    def curlparse(url):
        return urlparse(url)

    if "recoco" in args.input.lower():
        id_to_image = {
            img["id"]: {
                "image": img["file_name"].split("_")[-1],
                "height": img["height"],
                "width": img["width"]
            } for img in coco_data["images"]
        }
    elif "lvis" in args.input.lower():
        id_to_image = {
            img["id"]: {
                "image": curlparse(img["coco_url"]).path.lstrip("/"),
                "height": img["height"],
                "width": img["width"]
            } for img in coco_data["images"]
        }
    else:
        id_to_image = {
            img["id"]: {
                "image": img["file_name"],
                "height": img["height"],
                "width": img["width"]
            } for img in coco_data["images"]
        }

    # Group annotations per image
    annotations_by_image = defaultdict(list)
    for ann in coco_data["annotations"]:
        annotations_by_image[ann["image_id"]].append(ann)

    # ==== Convert to Qwen format ====
    qwen_entries = []

    for image_id, anns in annotations_by_image.items():
        img_info = id_to_image[image_id]
        width, height = img_info["width"], img_info["height"]
        filename = img_info["image"]

        conversations = []
        bboxes = []
        segmentations = []

        for ann in anns:
            category_name = category_id_to_name.get(ann["category_id"], "an object")
            category_name = category_name.replace('_', ' ')

            bboxes.append(normalize_bbox_to_poly(ann["bbox"], width, height))
            if "segmentation" in ann:
                if isinstance(ann["segmentation"], list) and isinstance(ann["segmentation"][0], dict):
                    # RLE format for refclef dataset
                    polygons = get_polygons_from_RLE(ann["segmentation"])
                else:
                    polygons = ann["segmentation"]
                polygons = [
                    poly + poly[:2] if poly[0] != poly[-2] or poly[1] != poly[-1] else poly
                    for poly in polygons ]
                segmentations.append(normalize_segmentation(polygons, width, height))

            conversations.append({
                "from": "human",
                "value": "<image>\nDescribe the object in <annotation>."
            })
            conversations.append({
                "from": "gpt",
                "value": ann.get("caption", f"The marked object appears to be {category_name}.")
            })

        qwen_entries.append({
            "id": image_id,
            "image": os.path.join(args.root, filename),
            "size": [height, width],
            "bbox": bboxes,
            "segmentation": segmentations,
            "conversations": conversations
        })

    # ==== Save ====
    with open(args.output, "w") as f:
        json.dump(qwen_entries, f, indent=2)


if __name__ == "__main__":
    main()
