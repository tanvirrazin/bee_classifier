export PYTHONPATH=/data/bhuiyan/bee_vs_nobee/models:/data/bhuiyan/bee_vs_nobee/models/research:/data/bhuiyan/bee_vs_nobee/models/research/slim
export PATH=${PATH}:$PYTHONPATH

rm -R inference_graph/*
~/.conda/envs/detect_waste/bin/python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_resnet_v2_atrous_coco_bumble_or_not.config --trained_checkpoint_prefix saved_models/bumble_or_not_faster_rcnn_inception_resnet_v2_atrous_coco_07_24_2020/model.ckpt-86 --output_directory inference_graph

# rm test.record
# ~/.conda/envs/detect_waste/bin/python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

rm test_detections.record
rm unseen_test_detections.record
~/.conda/envs/detect_waste/bin/python inference/infer_detections.py --input_tfrecord_paths=test.record --output_tfrecord_path=test_detections.tfrecord --inference_graph=inference_graph/frozen_inference_graph.pb
~/.conda/envs/detect_waste/bin/python inference/infer_detections.py --input_tfrecord_paths=unseen_test.record --output_tfrecord_path=unseen_test_detections.tfrecord --inference_graph=inference_graph/frozen_inference_graph.pb

~/.conda/envs/detect_waste/bin/python confusion_matrix.py --label_map=./training/bumble_or_not_label_map.pbtxt --detections_record=./test_detections.tfrecord --output_path=./confusion_matrix.csv
~/.conda/envs/detect_waste/bin/python confusion_matrix.py --label_map=./training/bumble_or_not_label_map.pbtxt --detections_record=./unseen_test_detections.tfrecord --output_path=./confusion_matrix.csv
