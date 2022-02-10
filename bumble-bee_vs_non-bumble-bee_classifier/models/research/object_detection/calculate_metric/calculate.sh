#! /bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -m beas

export PATH=$PATH:/share/apps/cuda/cuda-10.2/bin/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/cuda/cuda-10.2/lib64/
export CUDA_VISIBLE_DEVICES=0,1,2,3


export PYTHONPATH=/data/bhuiyan/bee_vs_nobee/models:/data/bhuiyan/bee_vs_nobee/models/research:/data/bhuiyan/bee_vs_nobee/models/research/slim
export PAlH=${PATH}:$PYTHONPATH

DATASET_NAME="unseen_new_gray"
INFERENCE_GRAPH_PATH=../inference_graph/frozen_inference_graph.pb

for FILENAME in "./../${DATASET_NAME}.record" "${INFERENCE_GRAPH_PATH}" "../training/bumble_or_not_label_map.pbtxt"
do
    if [ ! -f $FILENAME ]
    then
        echo "$0: File ${FILENAME} not found."
        exit 1
    fi
done

rm detections.record
~/.conda/envs/detect_waste/bin/python ../inference/infer_detections.py \
		--input_tfrecord_paths=../${DATASET_NAME}.record \
		--output_tfrecord_path=detections.record \
		--inference_graph=$INFERENCE_GRAPH_PATH

~/.conda/envs/detect_waste/bin/python confusion_matrix.py \
		--detections_record=detections.record \
 		--label_map=../training/bumble_or_not_label_map.pbtxt \
		--output_path=confusion_matrix.csv

rm -R input/ground-truth/*
~/.conda/envs/detect_waste/bin/python convert_gt_xml.py --dataset_name=$DATASET_NAME

rm input/detection-results/*.txt
~/.conda/envs/detect_waste/bin/python create_dt_txt.py \
		--dataset_name=$DATASET_NAME \
		--output_path=input/detection-results \
		--label_map_path=../training/bumble_or_not_label_map.pbtxt \
		--inference_graph_path=$INFERENCE_GRAPH_PATH

# for iou in 0.3 0.5 0.7 0.8 0.9
# for iou in 0.7
# do
#     echo -e "\nIoU=$iou"
#     ~/.conda/envs/detect_waste/bin/python main.py --set-class-iou bumblebee $iou not_bumblebee $iou
# done
