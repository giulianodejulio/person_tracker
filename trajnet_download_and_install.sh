mkdir trajnetpp
cd trajnetpp
# SETTING UP REPOSITORIES
## Clone Repositories
git clone https://github.com/vita-epfl/trajnetplusplusdataset.git
git clone https://github.com/vita-epfl/trajnetplusplusbaselines.git

## Make virtual environment
#virtualenv -p /usr/bin/python3.8 trajnetv
#source trajnetv/bin/activate

## Download Requirements
cd trajnetplusplusbaselines/ 
pip install -e .
cd ../trajnetplusplusdataset/ 
pip install -e .
pip install -e '.[test, plot]'
# DATASET PREPARATION
## Download Repository
wget https://github.com/sybrenstuvel/Python-RVO2/archive/master.zip
unzip master.zip
rm master.zip
## Setting up ORCA (steps provided in the Python-RVO2 repo)
cd Python-RVO2-main/
# pip install cmake  THIS MIGHT INTERFERE with /usr/bin/cmake
pip install cython
python3 setup.py build
sudo python3 setup.py install
cd ../
## Download Repository
wget https://github.com/svenkreiss/socialforce/archive/refs/heads/main.zip
unzip main.zip
rm main.zip
## Setting up Social Force
cd socialforce-main/
pip install -e .
cd $WORKSPACE_TO_SOURCE
# this last line is not present in the Trajnet tutorial. I use it for convenience.
