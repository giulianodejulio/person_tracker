cd trajnetpp

## Download Requirements
cd trajnetplusplusbaselines/ 
pip install -e .
cd ../trajnetplusplusdataset/ 
pip install -e .
pip install -e '.[test, plot]'
cd Python-RVO2-main/
# pip install cmake  THIS MIGHT INTERFERE with /usr/bin/cmake
pip install cython
python3 setup.py build
sudo python3 setup.py install
cd ../
## Setting up Social Force
cd socialforce-main/
pip install -e .
cd $WORKSPACE_TO_SOURCE
