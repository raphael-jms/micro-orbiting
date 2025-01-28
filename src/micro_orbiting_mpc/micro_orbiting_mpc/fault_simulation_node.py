#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

from px4_msgs.msg import ActuatorMotors

from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.util.utils import read_ros_parameter_file, ensure_proper_fault_information
from micro_orbiting_msgs.srv import SetActuatorFailure
from micro_orbiting_msgs.msg import FailedActuators, FailedActuator

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

        # Mapping between internal and gazebo indices
        self.idx_intern2gazebo = {
            0: 2,
            1: 3,
            2: 0,
            3: 1,
            4: 5,
            5: 4,
            6: 7,
            7: 6
        }

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
        self.control_signal_sub = self.create_subscription(
            ActuatorMotors,
            '/micro_orbiting/control_signal',
            self.control_signal_callback,
            qos_profile
        )

        self.add_failure_sub = self.create_subscription(
            FailedActuators,
            '/add_actuator_faults',
            self.add_failure_callback,
            qos_profile
        )

        self.full_control_publisher = self.create_publisher(
            ActuatorMotors,
            '/fmu/in/actuator_motors',
            qos_profile
        )

        self.fault_publisher = self.create_publisher(
            FailedActuators,
            '/micro_orbiting/failed_actuators',
            qos_profile
        )

        self.set_fault_with_controller = self.create_client(
            SetActuatorFailure,
            '/micro_orbiting/actuator_failure_internal'
        )

        self.timer = self.create_timer(1.0, self.publish_actuator_faults_callback)

    def publish_actuator_faults_callback(self):
        """Publish the actuator faults."""
        msg = FailedActuators()
        for failed_actuator in self.model.failed_actuators:
            act = FailedActuator()
            act.pos1 = failed_actuator["pos_ids"][0]
            act.pos2 = failed_actuator["pos_ids"][1]
            act.idx = failed_actuator["idx"]
            act.intensity = float(failed_actuator["intensity"])
            msg.failed_actuators.append(act)
        self.fault_publisher.publish(msg)

    def apply_initial_faults(self):
        """Apply faults from the config file."""
        if self.actuator_failures is None:
            return

        for fault in self.actuator_failures:
            self.model.add_actuator_fault(fault["act_ids"], fault["intensity"])

    def control_signal_callback(self, msg: ActuatorMotors):
        """Handle incoming control signals and pass on with added faults/failures."""
        # Create output message
        output_msg = ActuatorMotors()
        output_msg.timestamp = msg.timestamp
        output_msg.timestamp_sample = msg.timestamp_sample
        output_msg.reversible_flags = msg.reversible_flags
        
        # Initialize output control signals
        output_msg.control = msg.control 

        # Apply faults for failed actuators
        for failed_actuator in self.model.failed_actuators:
            int_idx = failed_actuator["idx"]
            gaz_idx = self.idx_intern2gazebo[int_idx]
            output_msg.control[gaz_idx] = failed_actuator["intensity"]

        # Publish the modified control signal
        self.full_control_publisher.publish(output_msg)

    def add_failure_callback(self, msg):
        """Subscriber to add new faults."""
        try:
            for failure in msg.failed_actuators:
                failure = ensure_proper_fault_information(failure, self.model)                   

                # add to model
                self.model.add_actuator_fault([failure.pos1, failure.pos2], failure.intensity)
            
            srv = SetActuatorFailure.Request()
            srv.failed_actuators = msg.failed_actuators

            def handle_response(future):
                try:
                    response = future.result()
                    if not response.success:
                        self.get_logger().error("Service call failed")
                except Exception as e:
                    self.get_logger().error(f"Service call error: {e}")

            future = self.set_fault_with_controller.call_async(srv)
            future.add_done_callback(handle_response)

        except Exception as e:
            self.get_logger().error(f"Failed to add fault: {e}")
        
def main(args=None):
    rclpy.init(args=args)
    node = FaultInjectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()