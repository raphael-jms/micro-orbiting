from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import SetEnvironmentVariable, ExecuteProcess

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

        # publish a trajectory once
        # ExecuteProcess(
        #     cmd=['ros2', 'topic', 'pub', '--once', '/trajectory_commands',
        #          'micro_orbiting_msgs/msg/TrajectoryMsg',
        #          '{"action": "hover", "duration": 30, "file_path": ""}'],
        # ),
        ExecuteProcess(
            cmd=['ros2', 'topic', 'pub', '--once', '/trajectory_commands',
                'micro_orbiting_msgs/msg/SetTrajectory',
                '{"action": "hover", "duration": 30, "file_path": ""}',
                '--qos-reliability', 'reliable'],
        ),
    ])