# hrii_person_tracker

The hrii_person_tracker package aims at simulating a person's trajectory tracking task using lidar sensing and an LSTM network.

## Create a new ROS workspace
Use `create_catkin_ws` to create a new ROS workspace, for example `person-tracker_ws`. Then, remove the build directories.
```bash
catkin clean -y
```

## Clone the repo
Clone the `hrii_person_tracker` repo in your ROS workspace:
```bash
cd $WORKSPACE_TO_SOURCE/src
git clone git@gitlab.iit.it:hrii/vision/skeleton-tracking/hrii_person_tracker.git
```

## Import the necessary HRII and external repositories
Import necessary HRII packages with:
```bash
cd $WORKSPACE_TO_SOURCE/src/hrii_person_tracker/config
git_import_repos person_tracker_repos.yaml
```
You will also need external repos:
```bash
cd $WORKSPACE_TO_SOURCE/src
git clone -b noetic git@github.com:giulianodejulio/laser_filtering.git
git clone -b noetic git@github.com:giulianodejulio/lstm_layers_array.git
git clone -b noetic git@github.com:giulianodejulio/people.git
git clone -b noetic git@github.com:giulianodejulio/hunav_sim.git
git clone -b noetic git@github.com:giulianodejulio/leg_tracker.git
git clone -b throttle-tf-repeated-data-error https://github.com/BadgerTechnologies/geometry2.git
```
`geometry2` is needed to avoid the terminal to be filled with `Warning: TF_REPEATED_DATA` warnings.

## Build the Docker image
<!-- Change working directory to `hrii_person_tracker` and build `person-tracker` Docker image by running: -->
Go to workspace directory and build the Docker image by running:
```bash
cd $WORKSPACE_TO_SOURCE
dhb --hrii_env full
```
<!-- cd $WORKSPACE_TO_SOURCE/src/hrii_person_tracker -->
<!-- dhb -f Dockerfile --hrii_env full -->
<!-- the `--hrii_env full` flag is used to install the HRII environment as well as matlogger2 and libfranka libraries. The resulting docker image will have the same name as the workspace. -->

## Run the Docker image
Next steps require to run the built image inside the Docker container using `dhr`:
```bash
dhr
```

## Install TrajNet++
```bash
cd src/hrii_person_tracker
. trajnet_download_and_install.sh
```
**Bug**: for now, the installation must be done every time the Docker container is run. Therefore, only the first time we need to run `trajnet_download_and_install.sh` to download and install Trajnet++. Once the directory `trajnetpp` gets created, next time we run the Docker image, we run the script `trajnet_install.sh`.

<!-- ## Matlogger2 and libfranka Install
Some packages depend on matlogger2 and libfranka libraries. To install them, run:
```bash
cd $WORKSPACE_TO_SOURCE/src/matlogger2/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=. -Dpybind11_DIR=$HOME/.local/lib/python3.8/site-packages/pybind11/share/cmake/pybind11 ..
make -j`nproc`
make install
sudo make install
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$WORKSPACE_TO_SOURCE/src/matlogger2/build"
cd $WORKSPACE_TO_SOURCE/src/libfranka/
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
``` -->

## First workspace build
Build the workspace for the first time and source it with
```bash
cd $WORKSPACE_TO_SOURCE
cb_full
source devel/setup.bash
```
<!-- # catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release $CMAKE_MATLOGGER2_DIR $CMAKE_LIBFRANKA_DIR -->
Next times we need to build the workspace, we can just run `catkin build`.

## Run the Gazebo scene
Finally, in one Docker terminal launch `roscore` and in a second the simulation in Gazebo with
```bash
mon launch hrii_person_tracker mobile_base_autonomous_navigation.launch 
```
<!-- and in the second one run
```bash
roslaunch hrii_leg_tracker legs_tracker.launch
``` -->

<!-- ## LSTM Prediction
Run the LSTM node with
```bash
rosrun hrii_person_tracker lstm_node.py
```
you can check topics `predicted_trajectory` for the output of the LSTM and `visualization_marker` to display the predicted trajectory in RViz.  -->