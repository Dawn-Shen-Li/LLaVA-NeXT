#!/bin/bash
## data preparation for Object Detection with VLM

# llmdet 
# ｜--huggingface
# ｜  |--bert-base-uncased
# ｜  |--siglip-so400m-patch14-384
# ｜  |--my_llava-onevision-qwen2-0.5b-ov-2
# ｜  |--mm_grounding_dino
# ｜  |  |--grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
# ｜  |  |--grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
# ｜  |  |--grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
# ｜--grounding_data
# ｜  |--coco
# ｜  |  |--annotations
# ｜  |  |  |--instances_train2017_vg_merged6.jsonl
# ｜  |  |  |--instances_val2017.json
# ｜  |  |  |--lvis_v1_minival_inserted_image_name.json
# ｜  |  |  |--lvis_od_val.json
# ｜  |  |--train2017
# ｜  |  |--val2017
# ｜  |--flickr30k_entities
# ｜  |  |--flickr_train_vg7.jsonl
# ｜  |  |--flickr30k_images
# ｜  |--gqa
# ｜  |  |--gqa_train_vg7.jsonl
# ｜  |  |--images
# ｜  |--llava_cap
# ｜  |  |--LLaVA-ReCap-558K_tag_box_vg7.jsonl
# ｜  |  |--images
# ｜  |--v3det
# ｜  |  |--annotations
# ｜  |  |  |--v3det_2023_v1_train_vg7.jsonl
# ｜  |  |--images
# ｜--LLMDet (code)

# .xdecoder_data
# └── coco/
#     ├── train2017/
#     ├── val2017/
#     ├── panoptic_train2017/
#     ├── panoptic_semseg_train2017/
#     ├── panoptic_val2017/
#     ├── panoptic_semseg_val2017/
#     └── annotations/
#         ├── refcocog_umd_val.json
#         ├── captions_val2014.json
#         ├── panoptic_val2017.json
#         ├── caption_class_similarity.pth
#         ├── panoptic_train2017_filtrefgumdval_filtvlp.json
#         ├── grounding_train2017_filtrefgumdval_filtvlp.json
#         └── coco_train2017_filtrefgumdval_lvis.json          # lvis dataset
 

# echo "Downloading coco dataset..."
mkdir -p coco

cd coco || exit 1
wget http://images.cocodataset.org/zips/train2017.zip -O train2017.zip
if [ $? -ne 0 ]; then
    echo "Failed to download train2017.zip."
    exit 1
fi
unzip train2017.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip train2017.zip."
    exit 1
fi
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
if [ $? -ne 0 ]; then
    echo "Failed to download val2017.zip."
    exit 1
fi
unzip val2017.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip val2017.zip."
    exit 1
fi

mkdir -p annotations
cd annotations || exit 1
wget https://huggingface.co/datasets/merve/coco/blob/main/annotations/instances_val2017.json
if [ $? -ne 0 ]; then
    echo "Failed to download instances_val2017.json."
    exit 1
fi

wget https://huggingface.co/datasets/merve/coco/blob/main/annotations/instances_train2017_vg_merged6.jsonl
if [ $? -ne 0 ]; then
    echo "Failed to download instances_train2017_vg_merged6.jsonl."
    exit 1
fi

#----------------------------------------

echo "Downloading lvis dataset..."
wget https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_v1_minival_inserted_image_name.json
if [ $? -ne 0 ]; then
    echo "Failed to download lvis_v1_minival_inserted_image_name.json."
    exit 1
fi

wget https://huggingface.co/GLIPModel/GLIP/resolve/main/lvis_od_val.json
if [ $? -ne 0 ]; then
    echo "Failed to download lvis_od_val.json."
    exit 1
fi  

#----------------------------------------
echo "Downloading GQA dataset..."
cd ../..
mkdir -p gqa
cd gqa || exit 1
wget https://nlp.stanford.edu/data/gqa/images.zip
if [ $? -ne 0 ]; then
    echo "Failed to download images.zip."
    exit 1
fi
unzip images.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip images.zip."
    exit 1
fi

#----------------------------------------
echo "Downloading llava_cap dataset..."
cd /data
mkdir -p llava_cap
cd llava_cap || exit 1
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
if [ $? -ne 0 ]; then
    echo "Failed to download images.zip."
    exit 1
fi
unzip images.zip
if [ $? -ne 0 ]; then
    echo "Failed to unzip images.zip."
    exit 1
fi   

#----------------------------------------
echo "Downloading flickr30K ..."
mkdir -p /data/flickr30k_entities
cd /data/flickr30k_entities || exit 1
wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr_annotations_30k.csv
wget https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main/flickr30k-images.zip
unzip flickr30k-images.zip -d flickr30k-images

#----------------------------------------
echo "Downloading V3Det dataset..."
mkdir -p /data/v3det
cd /data/v3det || exit 1
# openxlab login
# openxlab dataset get --dataset-repo V3Det/V3Det 
huggingface-cli download yhcao/V3Det_Backup --repo-type dataset --local-dir /data/v3det --local-dir-use-symlinks False --resume-download
echo "Downloading additional annotations for X-Decoder..."

#----------------------------------------
echo "Downloading xdecoder datasets..."
cd /data/coco/annotations || exit 1

wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/caption_class_similarity.pth
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/captions_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/grounding_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/panoptic_train2017_filtrefgumdval_filtvlp.json
wget https://huggingface.co/xdecoder/X-Decoder/resolve/main/refcocog_umd_val.json
wget https://raw.githubusercontent.com/peteanderson80/coco-caption/master/annotations/captions_val2014.json

wget https://huggingface.co/xdecoder/SEEM/resolve/main/coco_train2017_filtrefgumdval_lvis.json

#----------------------------------------
echo "Downloading refcoco datasets..."
wget https://web.archive.org/web/20220413011817/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip
unzip refclef.zip
unzip refcoco.zip
unzip refcoco+.zip
unzip refcocog.zip

cd /data/coco/images
wget https://web.archive.org/web/20220515000000/http://bvisionweb1.cs.unc.edu/licheng/referit/data/images/saiapr_tc-12.zip
unzip saiapr_tc-12.zip


# lvis https://huggingface.co/datasets/Voxel51/LVIS
pip install -U fiftyone