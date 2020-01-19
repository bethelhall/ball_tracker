#!/bin/bash
source ~/.virtualenvs/cv/bin/activate
# change environment right way, otherwise works on default env
python --version

models="COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
COCO-InstanceSegmentation/retinanet_R_101_FPN_3x.yaml
COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
"

lrs=(0.00025, 0.02)
iters=(200, 300, 800, 1000)

for i in $iters
do
    for l in $lrs
    do
        for m in $models
        do
            echo "training $m with lr: $lr & iter: $iter.........."
            echo "..........................."
            python ball_trainer.py --config-name $m --output ../data/output --opts SOLVER.MAX_ITER $i SOLVER.BASE_LR $l
        done
    done
done