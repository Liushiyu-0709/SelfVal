#!/bin/bash
export CUDA_VISIBLE_DEVICES="7"

PYTHON_PATH=/home/lsy/data/miniconda3/envs/selfval/bin/python

# model path dir (you can download in huggingface)
MODEL_PATH=/home/lsy/data/projects/LLaVA/checkpoints/llava-v1.5-7b
# output result path
OUTPUT_PATH=/home/lsy/data/projects/LLaVA-Align/experiments/out/test.jsonl
# sample nums
SAMPLE_NUM=3
# cache path
CACHE_PATH=/home/lsy/data/projects/LLaVA-Align/experiments/scripts/chair500/chair.pkl
# chair-500.jsonl file location
QUESTION_FILE=/home/lsy/data/mmdata/chair-500.jsonl
# CHAIR image dir
IMAGEDIR=/home/lsy/data/mmdata/chair-500




"$PYTHON_PATH" -m llava.eval.selection \
    --model-path $MODEL_PATH \
    --question-file $QUESTION_FILE \
    --answers-file $OUTPUT_PATH \
    --sample-num $SAMPLE_NUM \
    --extract-method mscoco \
    --cache $CACHE_PATH \
    --image-dir $IMAGEDIR
