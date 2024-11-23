from setuptools import setup

package_name = 'micro_orbiting_mpc'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
        ],
    },
)