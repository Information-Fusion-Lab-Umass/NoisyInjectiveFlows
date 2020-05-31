IMAGE_NAME=gpu_nf
docker build -t $IMAGE_NAME -f Dockerfile .
docker run --gpus all -it -v "$(pwd)":/app/host $IMAGE_NAME