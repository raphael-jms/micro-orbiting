# trajectory_init_node.py
import rclpy
from rclpy.node import Node
from micro_orbiting_msgs.msg import SetTrajectory

class TrajectoryInitNode(Node):
    def __init__(self):
        super().__init__('trajectory_init_node')
        
        # Declare the parameter with a default value
        self.declare_parameter('traj_shape', 'hover')
        
        # Create publisher
        self.publisher = self.create_publisher(
            SetTrajectory,
            '/trajectory_commands',
            10
        )
        
        # Publish once after a short delay to ensure everything is ready
        self.timer = self.create_timer(1.0, self.publish_trajectory)

    def publish_trajectory(self):
        msg = SetTrajectory()
        msg.action = self.get_parameter('traj_shape').value
        msg.duration = 30
        msg.file_path = ""
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published initial trajectory: {msg.action}')
        
        # Stop the timer after publishing
        self.timer.cancel()

def main():
    rclpy.init()
    node = TrajectoryInitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()