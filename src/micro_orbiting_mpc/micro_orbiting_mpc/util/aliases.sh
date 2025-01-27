# Useful commands for interacting with the ROS2 package.
# Source manually to command line or add `source /your_path/util/aliases.sh` to your bashrc file

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
alias viznode='ros2 run micro_orbiting_mpc viz_node'
alias vizplan='ros2 run micro_orbiting_mpc viz_plan'
alias viz="tmux new-session \; split-window \; send-keys -t 0 \"viznode\" C-m \; send-keys -t 1 \"plotj\" C-m \; select-pane -t 0 \; split-window -h \; send-keys \"vizplan\" C-m"
alias com="tmux new-session \; split-window \; send-keys -t 0 \"microdds\" C-m \; send-keys -t 1 \"qground\" C-m"

# manually publish actuator failures
pub_act_failure_idx() {
    if [[ $# -lt 2 ]] || [[ $(($# % 2)) -ne 0 ]]; then
        echo "Usage: publish_failed_actuator_idx idx1 intensity1 [idx2 intensity2 ...]"
        return 1
    fi
    
    local msg="{\"failed_actuators\": ["
    while [ $# -gt 0 ]; do
        local idx=$1
        local intensity=$2
        
        if ! [[ $idx =~ ^[0-7]$ ]]; then
            echo "Error: idx must be integer 0-7"
            return 1
        fi
        if ! [[ $intensity =~ ^[0-1](\.[0-9]+)?$ ]] || (( $(echo "$intensity > 1" | bc -l) )); then
            echo "Error: intensity must be float between 0-1"
            return 1
        fi
        msg+="{idx: $idx, intensity: $intensity}" # pos1 and pos2 will be 0 by default 
        shift 2
        if [ $# -gt 0 ]; then
            msg+=", "
        fi
    done
    msg+="]}"
    
    echo $msg
    ros2 topic pub --once /add_actuator_faults micro_orbiting_msgs/msg/FailedActuators "$msg"
}

pub_act_failure_pos() {
   if [[ $# -lt 3 ]] || [[ $(($# % 3)) -ne 0 ]]; then
       echo "Usage: publish_failed_actuator_pos pos1_1 pos2_1 intensity1 [pos1_2 pos2_2 intensity2 ...]" 
       return 1
   fi
   
   local msg="{\"failed_actuators\": ["
   while [ $# -gt 0 ]; do
       local pos1=$1
       local pos2=$2
       local intensity=$3
       
       if ! [[ $intensity =~ ^[0-1](\.[0-9]+)?$ ]] || (( $(echo "$intensity > 1" | bc -l) )); then
           echo "Error: intensity must be float between 0-1"
           return 1
       fi
       if ! [[ $pos1 =~ ^[1-4]$ ]] || ! [[ $pos2 =~ ^[1-2]$ ]]; then
           echo "Error: pos values must be integer pos1 in [1,4], pos2 in [1,2] is pos1 $pos1 and pos2 $pos2."
           return 1
       fi
       msg+="{intensity: $intensity, pos1: $pos1, pos2: $pos2}" # idx will be 0 by default 
       shift 3
       if [ $# -gt 0 ]; then
           msg+=", "
       fi
   done
   msg+="]}"
   
   echo $msg
   ros2 topic pub --once /add_actuator_faults micro_orbiting_msgs/msg/FailedActuators "$msg"
}