from PIL import Image, ImageDraw
import argparse
import json
import os

def draw_segmentation(ann, image_path):
    # Get the image and segmentation
    image = Image.open(image_path).copy()
    segmentation = ann['segmentation']  # usually a list of lists

    draw = ImageDraw.Draw(image)

    # If segmentation is a list of polygons, draw each
    if isinstance(segmentation[0], list):
        for poly in segmentation:
            draw.polygon(poly, outline="red", width=3)
    else:
        draw.polygon(segmentation, outline="red", width=3)

    image.save('./image.jpg')
    

def main():
    parser = argparse.ArgumentParser(description="Draw segmentation polygons on images.")
    parser.add_argument("json_path", type=str, help="Path to the annotation JSON file")
    parser.add_argument("image_root", type=str, help="Root directory containing images") #/data
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        data = json.load(f)

    
    print(json.dumps(data[1], indent=2))
    rec = data[1]
    
    image_path = os.path.join(args.image_root, rec['image'])
    height, width = rec['size'] # "size": [height, width],
    n = len(rec['bbox'])


    # Draw all bboxes in green
    for i in range(n):
        image = Image.open(image_path).copy()
        draw = ImageDraw.Draw(image)
        bbox = rec['bbox'][i]
        # bbox is [x_min, y_min, x_max, y_max] normalized [x / w, y / h, bw / w, bh / h]
        x_min = bbox[0] * width
        y_min = bbox[1] * height
        x_max = bbox[2] * width + x_min
        y_max = bbox[3] * height + y_min
        draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=3)
   
        seg = rec['segmentation'][i]
        # seg is a list of polygons, each polygon is a list of [x, y] normalized
        if isinstance(seg[0], list):
            for poly in seg:
                poly_abs = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(poly)]
                draw.polygon(poly_abs, outline="red", width=3)
        else:
            poly_abs = [coord * width if i % 2 == 0 else coord * height for i, coord in enumerate(seg)]
            draw.polygon(poly_abs, outline="red", width=3)

        image.save(f'./image_{i}.jpg')



if __name__ == "__main__":
    main()
    

# python data/verify_qwen_data.py xx.json /data