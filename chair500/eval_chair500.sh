CACHE_PATH=/home/lsy/data/projects/LLaVA-Align/selfval/chair500/chair.pkl
CAPFILE=/home/lsy/data/projects/LLaVA-Align/experiments/out/test.jsonl
ANNOTATION_PATH=/home/lsy/data/mmdata/MSCOCO/annotation/annotations

python ./chair500/chair.py \
--coco_path $ANNOTATION_PATH \
--cache $CACHE_PATH \
--cap_file $CAPFILE



