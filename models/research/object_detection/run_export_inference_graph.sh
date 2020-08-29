#!/bin/bash

export PYTHONPATH=/data/bhuiyan/bee_vs_nobee/models:/data/bhuiyan/bee_vs_nobee/models/research:/data/bhuiyan/bee_vs_nobee/models/research/slim
export PATH=${PATH}:$PYTHONPATH

python export_inference_graph.py --input_type image_tensor \
				 --pipeline_config_path training/faster_rcnn_inception_v2_bumble_or_not.config \
				 --trained_checkpoint_prefix saved_models/bumble_or_not_06_04_2020/model.ckpt-100000 \
				 --output_directory inference_graph
