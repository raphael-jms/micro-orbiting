from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare, Parameter
from launch.actions import SetEnvironmentVariable, ExecuteProcess, DeclareLaunchArgument

def generate_launch_description():
    # Declare a launch argument for the config file
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value='nominal_mpc.yaml',
        description='Name of the config file to use (must be in the config directory)'
    )

    # Create a path that uses the launch argument
    config = PathJoinSubstitution([
        FindPackageShare('micro_orbiting_mpc'),
        'config',
        LaunchConfiguration('config_file')
    ])
  
    return LaunchDescription([
        config_file_arg,  # Add the launch argument to the LaunchDescription
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
        Node(
            package='micro_orbiting_mpc',
            executable='spacecraft_mpc_node',
            name='spacecraft_mpc_node',
            parameters=[config],
            output='screen',
            prefix='python3 -u',
            additional_env={'PYTHONUNBUFFERED': '1'},
        ),
       
        Node(
            package='micro_orbiting_mpc',
            executable='trajectory_init_node',
            name='trajectory_init_node',
            parameters=[config],
        ),
    ])