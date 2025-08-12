import os
from datasets import load_dataset
from tqdm import tqdm
import json
from datasets import Features, Value, ClassLabel, Image, Sequence
'''
From: 
1. ================= mix665k (https://github.com/haotian-liu/LLaVA) in ./playground ====================
  Visual instruction tuning takes around 20 hours for LLaVA-v1.5-13B on 8x A100 (80G), 
  due to the increased resolution to 336px. It takes around 10 hours for LLaVA-v1.5-7B on 8x A100 (40G).
  Training script with DeepSpeed ZeRO-3: finetune.sh.

json: /home/shenli/Project/LLaVA-NeXT/playground/llava_v1_5_mix665k_v1.json

# organize the data as follows 
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
    
{
  'id': '000000033471', 
  'image': 'coco2017/images/train2017/000000033471.jpg', 
  'conversations': [
    {'from': 'human', 'value': '<image>\nWhat are the colors of the bus in the image?'}, 
    {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, 
    {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, 
    {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, 
    {'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, 
    {'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}]
}

2. ========================= cocodata ========
      dict_keys(['info', 'categories', 'annotations', 'images', 'licenses'])
json:{
  /data/coco/annotations/refclef/instances.json
  /data/coco/annotations/refcoco/instances.json
  /data/coco/annotations/refcoco+/instances.json
  /data/coco/annotations/refcocog/instances.json
  /data/coco/annotations/lvis/lvis_v1_train.json
}
>>> data['categories'][0]
{'image_count': 8, 'synonyms': ['aerosol_can', 'spray_can'], 'def': 'a dispenser that holds a substance under pressure', 'id': 1, 'synset': 'aerosol.n.02', 'name': 'aerosol_can', 'frequency': 'c', 'instance_count': 11}
>>> data['annotations'][0]
{'area': 289.97, 'id': 3, 'image_id': 76261, 'bbox': [238.16, 340.87, 66.44, 47.08], 'category_id': 99, 'segmentation': [[238.16, 386.84, 240.61, 387.95, 245.96, 385.27, 251.54, 381.48, 257.11, 377.69, 262.24, 373.67, 267.81, 370.55, 268.7, 372.56, 270.93, 374.12, 273.83, 372.78, 276.51, 369.66, 279.41, 369.88, 280.07, 367.42, 282.53, 365.42, 284.09, 362.07, 284.09, 359.39, 284.09, 356.94, 287.21, 355.38, 290.55, 353.59, 293.67, 351.36, 297.69, 348.01, 300.36, 345.56, 302.82, 342.66, 304.6, 340.87, 301.92, 340.87, 299.69, 344.0, 295.46, 346.9, 292.88, 348.73, 289.52, 350.97, 283.6, 354.81, 282.33, 353.53, 280.89, 354.65, 279.13, 357.69, 277.85, 359.29, 275.93, 362.17, 273.21, 362.97, 271.13, 364.25, 268.89, 366.34, 267.93, 367.94, 263.77, 370.18, 260.73, 372.26, 257.21, 374.66, 252.9, 377.7, 249.06, 380.58, 246.18, 382.66, 241.7, 383.94, 240.26, 385.22, 238.16, 386.84]]}
>>> data['images'][0]
{'date_captured': '2013-11-14 17:02:52', 'neg_category_ids': [279, 899, 127, 180, 1136, 725, 663], 'id': 397133, 'license': 4, 'height': 427, 'width': 640, 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'not_exhaustive_category_ids': [914, 801, 566, 139, 1021]}

3. ================== grounding xdecoder ===============
json:{
  /data/coco/annotations/xdecoder/grounding_train2017_filtrefgumdval_filtvlp.json
  #/data/coco/annotations/xdecoder/panoptic_train2017_filtrefgumdval_filtvlp.json
  /data/coco/annotations/xdecoder/refcocog_umd_val.json
}
>>> data.keys()
dict_keys(['images', 'annotations'])
>>> data['images'][0]
{'license': 1, 'file_name': '000000131074.jpg', 'coco_url': 'http://images.cocodataset.org/train2017/000000131074.jpg', 'height': 428, 'width': 640, 'date_captured': '2013-11-21 01:03:06', 'flickr_url': 'http://farm9.staticflickr.com/8308/7908210548_33e532d119_z.jpg', 'id': 131074}
>>> data['annotations'][0] 
{'segmentation': [[16.22, 84.86, 36.76, 52.43, 69.19, 45.95, 108.11, 50.27, 118.92, 60.0, 134.05, 76.22, 132.97, 104.32, 145.95, 128.11, 151.35, 143.24, 143.78, 164.86, 139.46, 174.59, 151.35, 191.89, 189.19, 223.24, 230.27, 265.41, 235.68, 268.65, 238.92, 282.7, 232.43, 297.84, 223.78, 302.16, 183.78, 290.27, 158.92, 271.89, 105.95, 251.35, 122.16, 272.97, 137.3, 287.03, 149.19, 312.97, 162.16, 338.92, 169.73, 349.73, 174.05, 361.62, 161.08, 396.22, 149.19, 423.24, 139.46, 437.3, 125.41, 448.11, 113.51, 453.51, 104.86, 454.59, 98.38, 444.86, 78.92, 426.49, 55.14, 402.7, 40.0, 387.57, 21.62, 368.11, 1.08, 348.65, 0.0, 136.76, 0.0, 123.78]], 'area': 57185.38189999998, 'iscrowd': 0, 'image_id': 519404, 'bbox': [0.0, 45.95, 238.92, 408.64], 'category_id': 1, 'id': 1241542, 'split': 'train', 'sentences': [{'tokens': ['two', 'woman', 'one', 'in', 'black', 'eatting', 'and', 'the', 'other', 'has', 'a', 'white', 'shirt', 'at', 'the', 'desk'], 'raw': 'Two woman one in black eatting and the other has a white shirt at the desk', 'sent_id': 0, 'sent': 'two woman one in black eatting and the other has a white shirt at the desk'}, {'tokens': ['woman', 'in', 'white', 'shirt', 'looking', 'down', 'at', 'laptop', 'computer'], 'raw': 'Woman in white shirt looking down at laptop computer.', 'sent_id': 1, 'sent': 'woman in white shirt looking down at laptop computer'}, {'tokens': ['woman', 'white', 'shirt'], 'raw': 'woman, white shirt', 'sent_id': 15378, 'sent': 'woman white shirt'}, {'tokens': ['white', 'shirt', 'lady'], 'raw': 'white shirt lady', 'sent_id': 15379, 'sent': 'white shirt lady'}, {'tokens': ['woman', 'in', 'white'], 'raw': 'woman in white', 'sent_id': 15430, 'sent': 'woman in white'}, {'tokens': ['blond', 'chick'], 'raw': 'blonde chick', 'sent_id': 15431, 'sent': 'blond chick'}, {'tokens': ['left', 'whitre', 'shirt'], 'raw': 'left whitre shirt', 'sent_id': 15432, 'sent': 'left whitre shirt'}], 'ann_id': 1241542}
dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id', 'split', 'sentences', 'ann_id'])

4. ================ llmdet ================  
jsonl : {
  /data/coco/annotations/llmdet/instances_train2017_vg_merged6.jsonl
  #/data/coco/annotations/llmdet/lvis_v1_minival_inserted_image_name.json
  /data/flickr30k/flickr_train_vg7.jsonl
  /data/gqa/gqa_train_vg7.jsonl
  /data/v3det/annotations/v3det_2023_v1_train_vg7.jsonl
  /data/llava_cap/LLaVA-ReCap-558K_tag_box_vg7.jsonl
}
only: bbox
 
>>> data[0]
{
  'filename': '000000272026.jpg', 'height': 480, 'width': 640, 
  'conversations': [
    {
      'from': 'human', 
      'value': '<image>\nProvide a scene graph caption of the given image.'
    }, 
    {
      'from': 'gpt', 
      'value': 'In the image, three women are sitting at a dining table. They appear to be enjoying themselves, with one leaning on the table and all of them in front of it. The table has several cups and bottles on it, suggesting that they are enjoying drinks. One of the women is beside a wall, and two of the women are beside each other. There is also a chair visible in the image, indicating that they are seated. The setting appears to be a restaurant or a similar dining establishment.'
    }
  ], 
  'tags': ['table', 'wall', 'caption', 'restaurant', 'dining establishment'], 'grounding': {
      'caption': 'keyboard. toaster. hair drier. tennis racket. bicycle. person. tv. cow. parking meter. couch. horse. cell phone. vase. sports ball. umbrella. airplane. apple. donut. suitcase. microwave. laptop. refrigerator. baseball bat. fork. orange. backpack. tie. sheep. bear. toothbrush. cup. motorcycle. snowboard. bed. bird. cake. sandwich. sink. teddy bear. skis. giraffe. bus. carrot. skateboard. potted plant. stop sign. dining table. wine glass. train. boat. bowl. surfboard. book. scissors. clock. fire hydrant. truck. bottle. toilet. spoon. mouse. car. traffic light. knife. kite. zebra. hot dog. elephant. cat. frisbee. baseball glove. pizza. handbag. chair. dog. bench. remote. oven. banana. broccoli. wall. restaurant. dining establishment. ', 
      'regions': [
        {
          'bbox': [[256.49, 255.39, 288.16, 348.46], [354.01, 287.67, 372.15, 347.32]], 
          'phrase': 'bottle', 
          'tokens_positive': [[515, 521]]
        }, 
        {
          'bbox': [224.36, 233.06, 526.38, 456.34000000000003], 
          'phrase': 'dining table', 
          'tokens_positive': [[415, 427]]
        }
      ]
    }
   }
dict_keys(['filename', 'height', 'width', 'conversations', 'tags', 'grounding']) 


To: 
=========== onevsions ==============
  {
    'id': '000000033471', 
    'image': 'coco/train2017/000000033471.jpg', 
    'conversations': [
      {'from': 'human', 'value': '<image>\nWhat are the colors of the bus in the image?'}, 
      {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, 
      {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, 
      {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, 
      {'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, 
      {'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}
    ]
  }
'''

json_path = "/home/shenli/Project/LLaVA-NeXT/playground/llava_v1_5_mix665k_v1.json"
with open(json_path, "r") as f:
    data = json.load(f)
# Fix types
for item in data:
    if not isinstance(item["id"], str):
        item["id"] = str(item.get("id", ""))  # Convert id to string
    if not isinstance(item.get("conversations"), list):
        item["conversations"] = []  # Default if missing or wrong

# Save to temp file
fixed_path = json_path.replace(".json", "_v1.json")
with open(fixed_path, "w") as f:
    json.dump(data, f, indent=2)
    
dataset = load_dataset(
    "json",
    data_files=json_path,
    split="train"
)

data = load_dataset("/data/huggingface/LLaVA-OneVision-Data/geo3k", split="train")
data = load_dataset("/data/huggingface/LLaVA-NeXT-Data", split="train") #  streaming=True)

image_folder = "<your_image_folder>"

converted_data = []

for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        json_data["image"] = f"{da['id']}.jpg"
        da["image"].save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)


with open("<your_json_file>.json", "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
    

# v3det 
dataset = load_dataset(
    "json",
    data_files="/data/v3det/annotations/v3det_2023_v1_train_vg7.jsonl",
    features=Features({
        "image": Image(),         # Automatically loads from path
        "label": Value("int64")   # or ClassLabel if you have text labels
    })
)

dataset = dataset.map(lambda x: {"image": f"image_path/{x['image']}"})