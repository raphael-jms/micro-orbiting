#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import ActuatorMotors

from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.util.utils import read_ros_parameter_file
from micro_orbiting_msgs.srv import SetActuatorFailure

class FaultInjectionNode(Node):
    """
    ROS2 node that acts as bridge between the controller and the robot node. Normally, the control 
    node would send the control inputs directly to the robot node. The FaultInjectionNode is added 
    in between, and emulates actuator failures by adding them to the control signal. 

    Add actuator failures in two ways:
    - At start up time: 
        - Read from the config YAML file (specified in the launch file)
        - Actuator failures that are present at start up time are always used
    - During run time: 
        - Using a ROS service
        - For safety reasons: These actuator failures are only active if the controller accepts them
    """
    def __init__(self):
        super().__init__('fault_injection_node')

        # QoS profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Initialize the model
        self.declare_parameter('time_step', 0.01)
        self.time_step = self.get_parameter('time_step').value
        robot_parameter_defaults = {
            "mass": 14.5, "inertia": 0.370, "max_force": 1.75, "thruster_distance_to_center": 0.14
        }
        self.robot_parameters = {}
        for param in robot_parameter_defaults.keys():
            self.declare_parameter(param, robot_parameter_defaults[param])
            self.robot_parameters[param] = self.get_parameter(param).value
        self.model = FreeFlyerDynamicsFull(self.time_step, self.robot_parameters)

        # Read initial faults from config
        self.declare_parameter('config_file', 'nominal_mpc.yaml')
        config_file = self.get_parameter('config_file').value
        self.actuator_failures = read_ros_parameter_file(config_file, 'actuator_failures')
        
        # Apply initial faults
        self.apply_initial_faults()

        # Create subscribers, publishers and services
        self.subscription = self.create_subscription(
            ActuatorMotors,
            '/micro_orbiting/control_signal',
            self.control_signal_callback,
            qos_profile
        )

        self.publisher = self.create_publisher(
            ActuatorMotors,
            '/fmu/in/actuator_motors',
            qos_profile
        )

        self.fault_service = self.create_service(
            SetActuatorFailure,
            '/add_actuator_faults',
            self.add_fault_callback
        )

    def apply_initial_faults(self):
        """Apply faults from the config file."""
        if self.actuator_failures is None:
            return

        for fault in self.actuator_failures:
            self.model.add_actuator_fault(fault["act_ids"], fault["intensity"])

    def control_signal_callback(self, msg: ActuatorMotors):
        """Handle incoming control signals and pass on with added faults/failures."""
        # Get the faulty signal (8-dimensional)
        _, faulty_input_full = self.model.get_faulty_input()
        
        # Create output message
        output_msg = ActuatorMotors()
        output_msg.timestamp = msg.timestamp
        output_msg.timestamp_sample = msg.timestamp_sample
        output_msg.reversible_flags = msg.reversible_flags
        
        # Initialize output control signals
        output_msg.control = msg.control 

        # Apply faults for failed actuators
        for failed_actuator in self.model.failed_actuators:
            idx = failed_actuator["idx"]
            output_msg.control[idx] = failed_actuator["intensity"]

        # Publish the modified control signal
        self.publisher.publish(output_msg)

    def add_fault_callback(self, request: SetActuatorFailure.Request, 
                          response: SetActuatorFailure.Response):
        """Service callback to add new faults."""
        try:
            self.model.add_actuator_fault(request.positions, request.intensity)
            response.success = True
        except Exception as e:
            self.get_logger().error(f"Failed to add fault: {e}")
            response.success = False
        
        return response

def main(args=None):
    rclpy.init(args=args)
    node = FaultInjectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()