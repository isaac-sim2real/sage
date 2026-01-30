#!/bin/bash

IMAGE_NAME="sage"
CONTAINER_NAME="sage_container"
HOST_DIR=$(pwd)
CONTAINER_DIR="/app"

if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    docker build -t $IMAGE_NAME .
fi

echo "------------------------------------------------------------------"
echo -e "\033[1;32mCONTAINER:\033[0m $CONTAINER_NAME"
echo "------------------------------------------------------------------"

xhost +local:docker > /dev/null

if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
        docker exec -it $CONTAINER_NAME bash
    else
        docker start -ai $CONTAINER_NAME
    fi
else

    docker run -it \
        --name=$CONTAINER_NAME \
        --entrypoint /bin/bash \
        --gpus all \
        --ipc=host \
        --net=host \
        --privileged \
        -e "PRIVACY_CONSENT=Y" \
        -e DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $HOME/.Xauthority:/root/.Xauthority \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/root/Documents:rw \
        -v $HOST_DIR:/$CONTAINER_DIR:rw \
        $IMAGE_NAME

fi

xhost -local:docker > /dev/null
