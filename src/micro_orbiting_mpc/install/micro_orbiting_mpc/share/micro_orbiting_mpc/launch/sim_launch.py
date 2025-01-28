from launch import LaunchDescription
from launch_ros.actions import Node, ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='micro_orbiting_mpc',
            namespace='spacecraft_mpc_node',
            executable='spacecraft_mpc_node',
            name='sim'
        ),
        # ExecuteProcess(cmd=['ping', '192.168.1.1'], output='screen'),
        # Start PX4/ROS Interface: ~/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent udp4 -p 8888
        ExecuteProcess(cmd=['~/Micro-XRCE-DDS-Agent/build/MicroXRCEAgent', 'udp4'], p='8888'),
        # Start QGroundControl:
        ExecuteProcess(cmd=['~/QGroundControl/qgroundcontrol/build/QGroundControl'])
    ])