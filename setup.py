from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=[
        'trajnetbaselines',
        'trajnetbaselines.lstm'
    ],
    package_dir={
        '': 'trajnetpp/trajnetplusplusbaselines'},
)

setup(**setup_args)