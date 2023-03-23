#!/bin/bash

echo -e '\n===================================================================\n    RECUERDA SELECCIONAR LAS GPUS QUE QUIERES QUE SEAN VISIBLES\n             export CUDA_DEVICE_ORDER="PCI_BUS_ID"\n             export CUDA_VISIBLE_DEVICES="0,1"\n===================================================================\n'

IMAGE_NAME="siammt"

# Run container (Si utilizo "-u $(id -u):$(id -g)", no me forwardea bien la imagen)
# El "--cap-add SYS_ADMIN --device /dev/fuse" es para poder usar sshfs
docker run --gpus all -u $(id -u):$(id -g) -ti --rm \
    --name="SiamMT" \
    -v /home/lorenzo.vaquero/SiamMOTION/:/home/lorenzo.vaquero/SiamMOTION/ \
    -v /mnt/media/Lorenzo/:/home/lorenzo.vaquero/PHD/Benchmarks/HDD/ \
    -v /mnt/nfs/SiamMT_datasets/:/home/lorenzo.vaquero/PHD/Benchmarks/NFS/ \
    --net=host --env="DISPLAY" --volume="$HOME/.Xauthority:/root/.Xauthority:rw" \
    --cap-add SYS_ADMIN --device /dev/fuse \
    ${IMAGE_NAME}
