# Failsafe control of space robotics

This repo provides a ROS2 implementation of controllers for failsafe control of spacecraft. The controllers are based on the Master's thesis "Failsafe Control for Space Robotic Systems - Model Predictive Control under Actuator Failures"[XXXlink].

The repo provides
- MPC controller nodes 
    - for fault-free mode
    - for mode under actuator failures
    - fallback controller that initializes a new controller if actuator failures appear
- Real-time visualization node of the actuator forces and the previous path
- Real-time visualization of the planned path
- Presets for [PlotJuggler](https://github.com/facontidavide/PlotJuggler) for visualization of other trajectories and data 

The code can be tested in a hardware-in-the-loop simulation using [PX4/Gazebo](https://github.com/DISCOWER/PX4-Space-Systems). Installation instructions are provided [here](installation.md).

## Results

Implementation on a physical system, made at the KTH Space Robotics Lab (see more about this [here](https://discower.io/ "Discower project"))

Example: Failure of 3 thrusters

[![Video of a successful recovery after thruster failure](https://raphaelstockner.com/assets/failsafe_robotics.mp4)](https://raphaelstockner.com/assets/failsafe_robotics.mp4)

## Cite

Stöckner, R. (2024). _Failsafe Control for Space Robotic Systems - Model Predictive Control under Actuator Failures_ [Master's thesis, Kungliga Tekniska Högskolan]





The code is stable, but the repo is not yet completely cleaned up, this will happen during the next weeks.

