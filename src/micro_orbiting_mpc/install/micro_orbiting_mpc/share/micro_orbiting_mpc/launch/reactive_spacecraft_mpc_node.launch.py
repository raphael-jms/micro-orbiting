from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare, Parameter
from launch.actions import SetEnvironmentVariable, ExecuteProcess, DeclareLaunchArgument

def generate_launch_description():
    # Declare a launch argument for the controller config file
    config_file_arg = DeclareLaunchArgument(
        'config_nominal',
        default_value='nominal_mpc.yaml',
        description='Name of the config file to use (must be in the config directory)'
    )

    # Create a path that uses the launch argument
    config = PathJoinSubstitution([
        FindPackageShare('micro_orbiting_mpc'),
        'config',
        LaunchConfiguration('config_file')
    ])

     # Declare a launch argument for the robot parameter file
    robot_parameters_arg = DeclareLaunchArgument(
        'robot_parameters',
        default_value='robot_parameters_gazebo.yaml',
        description='Name of the robot parameter file to use (must be in the config directory)'
    )

    robot_params = PathJoinSubstitution([
        FindPackageShare('micro_orbiting_mpc'),
        'config',
        LaunchConfiguration('robot_parameters')
    ])

    return LaunchDescription([
        config_file_arg,
        robot_parameters_arg,
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
        Node(
            package='micro_orbiting_mpc',
            executable='spacecraft_mpc_node',
            name='spacecraft_mpc_node',
            parameters=[
                config,
                robot_params,
                {'config_file': LaunchConfiguration('config_file')}
            ],
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

        Node
        (
            package='micro_orbiting_mpc',
            executable='fault_simulation_node',
            name='fault_simulation_node',
            parameters=[
                config,
                robot_params,
                {'config_file': LaunchConfiguration('config_file')}
            ],
            output='screen',
            prefix='python3 -u',
            additional_env={'PYTHONUNBUFFERED': '1'},
        ),
    ])