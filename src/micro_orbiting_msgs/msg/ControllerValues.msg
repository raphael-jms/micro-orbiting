# ControllerValues.msg

# Header for timestamping
std_msgs/Header header

# State variables (x)
float64 x1
float64 y1
float64 alpha
float64 x2
float64 y2
float64 omega

# Error terms (e)
float64 e1
float64 e2
float64 e_alpha
float64 e3
float64 e5
float64 e_omega

# Control inputs (u)
float64[8] u
float64[8] u_nom
float64[8] u_control

# Planned state trajectory
float64[] plan_x1
float64[] plan_y1
float64[] plan_alpha
float64[] plan_x2
float64[] plan_y2
float64[] plan_omega

# Performance metrics
float64 control_cost
float64 solver_time
int32 solver_state

# For reference, the u arrays are organized as:
# u[0] = u11, u[1] = u12
# u[2] = u21, u[3] = u22
# u[4] = u31, u[5] = u32
# u[6] = u41, u[7] = u42
# (Same pattern for u_nom and u_control)