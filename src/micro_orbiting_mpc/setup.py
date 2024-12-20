import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'micro_orbiting_mpc'

setup(
    name=package_name,
    version='0.0.1',
    # packages=find_packages(),
    # package_dir={'': 'micro_orbiting_mpc'}, 
    # packages=find_packages(where="micro_orbiting_mpc"),
    # packages=find_packages(where=f'{package_name}*', include="*"), 
    # packages=find_packages(include=[package_name]), 
    packages=find_packages(), 
    # packages=["models", "controllers", "util"],
    # packages=[package_name, "models", "controllers", "util"],
    # packages=[package_name, package_name+"/models", package_name+"/controllers", package_name+"/util"],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        # (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Raphael Stöckner',
    maintainer_email='stockner@kth.se',
    description='Implementation of Thesis Failsafe Control for Space Robotic Systems Model Predictive Control under Actuator Failures',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'spacecraft_mpc_node = micro_orbiting_mpc.spacecraft_mpc_node:main'
            # 'spacecraft_mpc_node = spacecraft_mpc_node:main'
        ],
    },
)
