Pytracking Docker environment
=========================

Dockerfile bulding the pytracking environment, together with a non-root user.
Includes also all Torch dependencies which can be exploited using NVIDIA Container Toolkit.

### Building the image
This builds the pytracking image and sets up a user in the container with the same UID as the current host user and password "password".
```bash
./build_docker.sh
```

### Running the docker
```bash
docker run -it --gpus all -v /home/${USER}/.Xauthority:/home/${USER}/.Xauthority -e DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /home/${USER}/elia_franc_329:/home/${USER}/code -v /home/${USER}/dataset_processing/drone_dataset_split:/home/${USER}/drone_dataset_split -v /datasets/EV_UAV2:/home/${USER}/EV_UAV2 --rm --net=host pytracking_${USER} bash

```
