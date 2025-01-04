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

    # Fallback if not found in parameters
    traj_shape_arg = DeclareLaunchArgument(
        'traj_shape',
        default_value='hover',
        description='Shape of the trajectory to follow'   
    )
   
    return LaunchDescription([
        SetEnvironmentVariable('PYTHONUNBUFFERED', '1'),
        traj_shape_arg,
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

        # ExecuteProcess(
        #     cmd=[
        #         'ros2', 'topic', 'pub', '--once', '/trajectory_commands',
        #         'micro_orbiting_msgs/msg/SetTrajectory',
        #         [
        #             '{action: "', LaunchConfiguration('traj_shape'), '", duration: 30, file_path: ""}'
        #         ],
        #         '--qos-reliability', 'reliable'
        #     ],
        # ),
        
        Node(
            package='micro_orbiting_mpc',
            executable='trajectory_init_node',
            name='trajectory_init_node',
            parameters=[config],  # Use the same config file
            output='screen',
        ),
    ])