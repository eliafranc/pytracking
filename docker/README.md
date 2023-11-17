# Pytracking Docker environment

Dockerfile bulding the pytracking environment, together with a non-root user.
Includes also all Torch dependencies which can be exploited using NVIDIA Container Toolkit.

### Building the image

This builds the pytracking image and sets up a user in the container with the same UID as the current host user and password "password".

```bash
./build_docker.sh
```

### Running the docker

Depending on what kind of directories should be mounted, the _docker run_ command should be adjusted. Any mount can be specified with the -v flag. To just mount the pytracking code run the following command from the directory the repository root is located:

```bash
docker run -it --gpus all -v /home/${USER}/.Xauthority:/home/${USER}/.Xauthority -e DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v ${PWD}/pytracking/:/home/${USER}/pytracking --rm --net=host pytracking_${USER} bash

```
