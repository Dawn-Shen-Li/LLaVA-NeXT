import json, os
import numpy 
import argparse
from collections import defaultdict
from urllib.parse import urlparse
import cv2
from pycocotools import mask as mask_utils

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COCO annotations to Qwen1.5 format for visual_prompting.")
    parser.add_argument("input", help="Path to COCO annotation JSON file.")
    parser.add_argument("output", help="Output JSON file path.")
    parser.add_argument("--root", default="", help="Directory containing images referenced in the COCO annotations.")
    return parser.parse_args()

def normalize_bbox(bbox, w, h):
    x, y, bw, bh = bbox
    return [x / w, y / h, bw / w, bh / h]

def flatten_bbox(bboxes):
    # flatten list of bboxes: [[x1,y1,x2,y2], [x1,y1,x2,y2], ...] -> [x1,y1,x2,y2, x1,y1,x2,y2,...]
    flat = []
    for b in bboxes:
        flat.extend(b)
    return flat

def main():
    args = parse_args()
    # ==== Load ====
  
    with open(args.input, "r") as f:
        for line in f:
            data = json.loads(line)
  
            width = data['width']
            height = data['height']
            filename = data['filename']            
            # Normalize all bboxes, flatten
            bboxes = []
            phrases = []
            for region in data['grounding']['regions']:
                # region bbox could be list of lists or list of floats
                bbox_raw = region['bbox']
                # Some bbox are nested lists (polygons?), choose flatten or approx with bounding rect?
                # For simplicity, if bbox is nested, flatten and normalize each point:
                if isinstance(bbox_raw, list):  # polygon-like bbox
                    # normalize each point, flatten all points                    
                    for bbox in bbox_raw:
                        bboxes.append(normalize_bbox(bbox, width, height))
                        phrases.append(region.get('phrase', ''))
                else:
                    # plain bbox [x1, y1, x2, y2]
                    bboxes.append(normalize_bbox(bbox_raw, width, height))
                    phrases.append(region.get('phrase', ''))
            

            # Prepare conversations list:
            conversations = []
            for phrase in phrases:
                conversations.append({
                    "from": "human",
                    "value": "<image>\nDescribe the object in <annotation>."
                })
                conversations.append({
                    "from": "gpt",
                    "value": f"The marked object appears to be {phrase}."
                })

            qwen_entries.append({              
                "image": os.path.join(args.root, filename),
                "size": [height, width],
                "bbox": bboxes,
                "conversations": conversations
            })
            
    # ==== Save ====
    with open(args.output, "w") as f:
        json.dump(qwen_entries, f, indent=2)

if __name__ == "__main__":    
    # Example usage with your sample data dictionary as `data`
    main()
