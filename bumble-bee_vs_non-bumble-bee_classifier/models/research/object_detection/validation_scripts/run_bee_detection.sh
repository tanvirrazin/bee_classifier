export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_VISIBLE_DEVICES=0,1,2,3

export PYTHONPATH=/home/gpu-machine/Documents/tanvir/bumblebee_backend/models:/home/gpu-machine/Documents/tanvir/bumblebee_backend/models/research:/home/gpu-machine/Documents/tanvir/bumblebee_backend/models/research/slim
export PATH=${PATH}:$PYTHONPATH

# /home/gpu-machine/anaconda3/envs/tanvir_env/bin/python bee_testing_object_detection_image.py
/home/gpu-machine/anaconda3/envs/tanvir_env/bin/python mimic_bee_validation.py
