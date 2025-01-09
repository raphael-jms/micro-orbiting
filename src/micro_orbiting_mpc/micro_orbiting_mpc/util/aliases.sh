# General ROS2 shortcuts
alias r2tl='ros2 topic list'
alias r2nl='ros2 node list'
function rdid() {
    export ROS_DOMAIN_ID=$1
    echo "ROS_DOMAIN_ID set to $ROS_DOMAIN_ID"
}
function r2te() {
    ros2 topic echo $1
}
function r2thz() {
    ros2 topic hz $1
}
alias plotj='ros2 run plotjuggler plotjuggler -n'

# Micro Orbiting MPC shortcuts
function bot_hover() {
    # Hover at x, y, alpha
    ros2 topic pub --once /trajectory_commands micro_orbiting_msgs/msg/SetTrajectory "{action: 'hover_$1_$2_$3', duration: 5, file_path: ''}"
}