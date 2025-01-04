from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare, Parameter
from launch.actions import SetEnvironmentVariable, ExecuteProcess, DeclareLaunchArgument

def generate_launch_description():
    config = PathJoinSubstitution([
        FindPackageShare('micro_orbiting_mpc'),
        'config',
        'nominal_mpc.yaml'
    ])
  
    return LaunchDescription([
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
        Node(
            package='micro_orbiting_mpc',
            executable='spacecraft_mpc_node',
            name='spacecraft_mpc_node',
            parameters=[config],
            output='screen',
            prefix='python3 -u',  # Forces unbuffered output: actually shows output directly
            emulate_tty=True,
            additional_env={'PYTHONUNBUFFERED': '1'}, # should also force unbuffered output
        ),
       
        Node(
            package='micro_orbiting_mpc',
            executable='trajectory_init_node',
            name='trajectory_init_node',
            parameters=[config],  # Use the same config file
            output='screen',
        ),
    ])