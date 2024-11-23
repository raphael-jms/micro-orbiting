class States:
    """
    The constants describe how the thrusters of one F_{i} behave and are defined in the 
    following way:
    OPPOSING_FREE: Both actuators F_{i,1} and F_{i,2} are free
    FIRST_STUCK: The first actuator F_{i,1} is stuck, F_{i,2} is free
    SECOND_STUCK: The second actuator F_{i,2} is stuck, F_{i,1} is free
    BOTH_STUCK: Both actuators F_{i,1} and F_{i,2} are stuck
    """
    INVALID = -1 # i.e. blocking state has not been calculated yet
    OPPOSING_FREE = 1
    FIRST_STUCK = 2
    SECOND_STUCK = 3
    BOTH_STUCK = 4

    """
    These constants describe the rough fault category that occured
    """
    ORIGIN_IN_INTERIOR_OF_U = 10
    ORIGIN_ON_BOUNDARY_OF_U  = 11
    ORIGIN_OUTSIDE_OF_U  = 12

