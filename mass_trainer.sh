#!/bin/bash
source ~/.virtualenvs/cv/bin/activate
# change environment right way, otherwise works on default env
python --version

models="COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml
COCO-Detection/retinanet_R_101_FPN_3x.yaml
COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml
"

lrs=(0.02 0.002)
iters=(125 150 175 200)

m="COCO-Detection/retinanet_R_101_FPN_3x.yaml"  # specific model

for i in "${iters[@]}"
do
    for l in "${lrs[@]}"
    do
        # for m in $models
        # do
            echo "training ${m} with lr: ${l} & iter: ${i}.........."
            echo "..........................."
            python ball_trainer.py --config-name $m --output ../data/ball_output --opts SOLVER.MAX_ITER $i SOLVER.BASE_LR $l
        # done
    done
done
