#!/bin/bash
repos=(
    "coco/annotations/lvis/lvis_v1_train.json" "coco2017/images"
    "coco/annotations/lvis/lvis_v1_val.json" "coco2017/images"
    "/data/coco/annotations/refclef/instances.json" "coco/saiapr_tc-12"
    "/data/coco/annotations/refcoco/instances.json" "coco2017/images/train2017"
)

for ((i=0; i<${#repos[@]}; i+=2)); do
    input_file="${repos[i]}"
    root_dir="${repos[i+1]}"
    output_file="${input_file%.json}_qwen1.5_vp.json"
    echo "python coco2qwen.py $input_file $output_file --root $root_dir"
    python coco2qwen.py "$input_file" "$output_file" --root "$root_dir"
done