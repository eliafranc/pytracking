Pytracking
=========================

Slightly modified version of the [pytracking](https://github.com/visionml/pytracking) repository.
In order to use use any of the trackers, it is necessary to build the [pytracking-docker](../pytracking-docker).
Follow the steps described in the [README.md](../pytracking-docker/README.md) and find the mounted project repository in the docker root directory, once it is running.

### Before Running a Tracker
To finalize the setup of pytracking make sure to run the [install script](install.sh). The install script downloads the 
wished network models for the trackers. Per default, the networks associated with the RTS tracker are downloaded. 
The install script also sets up the environment by running the *create_default_local_file()* in [pytracking](pytracking/evaluation/environment.py) and [lts](ltr/admin/environment.py) which create *local.py* files which describe where the pretrained networks are stored for example.

```bash
./install.sh
```

After running the install script, make sure that the paths in the *local.py* files for [pytracking](pytracking/evaluation/local.py) and [ltr](ltr/admin/local.py) are set correctly, according to your environment.


### Running the RTS tracker on video data
The nice thing about the RTS tracker is that one can interactively draw a bounding box around the object one wants to
track while the tracker is running. In order to do so, navigate to the pytracking directory and run *run_python*.py.

```bash
cd pytracking
python3 run_video.py rts rts50 <path-to-video> --debug=<debug-level>

```