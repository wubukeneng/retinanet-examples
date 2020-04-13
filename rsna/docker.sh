sudo docker run --gpus all --ipc=host -v /home/xd/data/rsna/:/workspace/data/ -v /home/xd/project/rsna/:/workspace/rsna/ -v /home/xd/project/retinanet-examples/:/workspace/retinanet/ -dit odtk:latest
