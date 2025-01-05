import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from micro_orbiting_msgs.msg import SetTrajectory

class TrajectoryInitNode(Node):
    def __init__(self):
        super().__init__('trajectory_init_node')
        
        # Declare the parameter with a default value
        self.declare_parameter('traj_shape', 'hover')

        # Create QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publisher with QoS profile
        self.publisher = self.create_publisher(
            SetTrajectory,
            '/trajectory_commands',
            qos_profile
        )

        while self.count_subscribers('/trajectory_commands') == 0:
            self.get_logger().info('Waiting for subscribers...')
            rclpy.spin_once(self, timeout_sec=1.0)
        
        # Publish immediately
        self.publish_trajectory()

        # self.timer = self.create_timer(1, self.cmdloop_callback)

    def publish_trajectory(self):
        msg = SetTrajectory()
        msg.action = self.get_parameter('traj_shape').value
        msg.duration = 30
        msg.file_path = ""

        self.publisher.publish(msg)
        self.get_logger().info(f'Published initial trajectory: {msg.action}')

    def cmdloop_callback(self):
        self.publish_trajectory() 

def main():
    rclpy.init()
    node = TrajectoryInitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()