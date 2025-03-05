# Failsafe control of space robotics

This repo provides a ROS2 implementation of controllers for failsafe control of spacecraft. The controllers are based on the Master's thesis "Failsafe Control for Space Robotic Systems - Model Predictive Control under Actuator Failures" ([read here](https://raphaelstockner.com/assets/Failsafe%20Control%20for%20Space%20Robotic%20Systems%20-%20Model%20Predictive%20Control%20under%20Actuator%20Failures.pdf "Download PDF")).

The repo provides
- MPC controller nodes 
    - for fault-free mode
    - for mode under actuator failures
    - fallback controller that initializes a new controller if actuator failures appear
- Real-time visualization node of the actuator forces and the previous path
- Real-time visualization of the planned path
- Presets for [PlotJuggler](https://github.com/facontidavide/PlotJuggler) for visualization of other trajectories and data 

The code can be tested in a hardware-in-the-loop simulation using [PX4/Gazebo](https://github.com/DISCOWER/PX4-Space-Systems). Installation instructions are provided [here](Installation.md).

## Results

The implementation was on a physical system at the KTH Space Robotics Lab (learn more about this [here](https://discower.io/ "Discower project"))

Example: Recovery after the failure of 3 thrusters

https://github.com/user-attachments/assets/1233ae1e-a9e8-4787-b77b-609ad5c37aaf

## Cite

Stöckner, R. (2024). _Failsafe Control for Space Robotic Systems - Model Predictive Control under Actuator Failures_ [Master's thesis, Kungliga Tekniska Högskolan]

