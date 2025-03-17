# ROS NOETIC IMAGE
FROM osrf/ros:noetic-desktop-full

# SET BASH SHELL
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

# UPDATE PACKAGE LISTS, INSTALL BASIC SYSTEM UTILITIES
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-catkin-tools \
    # Useful tools
    gedit \
    mlocate \
    ipython3 \
    tree \
    wget \
    unzip \
    psmisc \
    ros-$ROS_DISTRO-rosmon \
    #Python modules
    python3-tk \
    python3-virtualenv \
# ROS dependencies needed by package https://github.com/wg-perception/people/tree/noetic/leg_detector
    liborocos-bfl-dev \
    ros-$ROS_DISTRO-easy-markers \
    ros-$ROS_DISTRO-kalman-filter \
    # ros-$ROS_DISTRO-map-laser \
# ROS dependencies needed by the ROS packages of the ROS workspace (found using rosdep)
    ros-$ROS_DISTRO-joint-trajectory-controller \
    ros-$ROS_DISTRO-move-base \
    ros-$ROS_DISTRO-global-planner \
    ros-$ROS_DISTRO-teb-local-planner \
    ros-$ROS_DISTRO-ira-laser-tools \
    ros-$ROS_DISTRO-amcl \
    ros-$ROS_DISTRO-map-server \
    ros-$ROS_DISTRO-gmapping \
    ros-$ROS_DISTRO-hector-gazebo-plugins \
    ros-$ROS_DISTRO-pinocchio \
    ros-$ROS_DISTRO-effort-controllers \
    ros-$ROS_DISTRO-velocity-controllers \
    ros-$ROS_DISTRO-industrial-robot-status-interface \
    ros-$ROS_DISTRO-force-torque-sensor-controller \
    ros-$ROS_DISTRO-industrial-robot-status-controller \
    ros-$ROS_DISTRO-teleop-twist-keyboard \
    ros-$ROS_DISTRO-twist-mux \
    socat

# SCIKIT-LEARN (HRII_LEG_TRACKER DEPENDENCY), LAST VERSION OF NUMPY
RUN pip3 install --no-cache-dir scikit-learn \
                                numpy==1.24.4 \
# TRAJNET++ DEPENDENCIES
                                pysparkling \
                                typing-extensions \
                                tqdm \
                                python-dateutil \
                                pytz \
                                pykalman \
                                pandas \
                                python-json-logger \
                                tzdata \
                                pylint \
                                pytest \
                                dill \
                                isort \
                                tomli \
                                platformdirs \
                                astroid \
                                tomlkit \
                                mccabe \
                                exceptiongroup \
                                pluggy \
                                iniconfig \
                                packaging \
                                torch==1.10.0