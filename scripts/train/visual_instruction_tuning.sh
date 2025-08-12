
#############Visual Instruction Tuning############
#Please download the annotation of the final mixture our instruction tuning data llava_v1_5_mix665k.json, and download the images from constituting datasets:
#   mix665k
#   {
#     'id': '000000033471', 
#     'image': 'coco/train2017/000000033471.jpg', 
#     'conversations': [
#       {'from': 'human', 'value': '<image>\nWhat are the colors of the bus in the image?'}, 
#       {'from': 'gpt', 'value': 'The bus in the image is white and red.'}, 
#       {'from': 'human', 'value': 'What feature can be seen on the back of the bus?'}, 
#       {'from': 'gpt', 'value': 'The back of the bus features an advertisement.'}, 
#       {'from': 'human', 'value': 'Is the bus driving down the street or pulled off to the side?'}, 
#       {'from': 'gpt', 'value': 'The bus is driving down the street, which is crowded with people and other vehicles.'}
#     ]
#   }
# ├── coco
# │   └── train2017
# ├── gqa
# │   └── images
# ├── ocr_vqa
# │   └── images
# ├── textvqa
# │   └── train_images
# └── vg
#     ├── VG_100K
#     └── VG_100K_2

# New options to note:

# --mm_projector_type mlp2x_gelu: the two-layer MLP vision-language connector.
# --vision_tower openai/clip-vit-large-patch14-336: CLIP ViT-L/14 336px.
# --image_aspect_ratio pad: this pads the non-square images to square, instead of cropping them; it slightly reduces hallucination.
# --group_by_modality_length True: this should only be used when your instruction tuning dataset contains both language (e.g. ShareGPT) and multimodal (e.g. LLaVA-Instruct). It makes the training sampler only sample a single modality (either image or language) during training, which we observe to speed up training by ~25%, and does not affect the final outcome.


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-13b-v1.5 \
    --version v1 \
    --data_path ./playground/data/llava_v1_5_mix665k.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-13b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-13b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb