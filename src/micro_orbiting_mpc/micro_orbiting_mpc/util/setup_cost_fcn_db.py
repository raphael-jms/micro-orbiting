import argparse
import pprint

from micro_orbiting_mpc.controllers.spiralMPC_eMPC.explicit_mpc_terminal_incredients import explicitMPCTerminalIngredients
from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.util.cost_handler import CostHandler
from micro_orbiting_mpc.util.yes_no_question import query_yes_no
from micro_orbiting_mpc.util.utils import read_ros_parameter_file

def calculate_cost_fcn(model, cost_handler, tuning):
    # TODO include the tuning in the database


    # Create the terminal cost function
    terminal_cost = explicitMPCTerminalIngredients(model, tuning)

    terminal_cost.calculate_terminal_ingredients()

    tcost = terminal_cost.get_terminal_cost()
    tset = terminal_cost.get_terminal_set()

    # Save the terminal cost function to the database
    cost_handler.set_cost_fcn(tcost, tset, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for setting up the term cost database")

    parser.add_argument('--overwrite', action='store_true', help='Overwrite the existing database')
    parser.add_argument('--resolution', type=int, default=4, help='Number of cases considered per thruster')
    parser.add_argument('--db_name', type=str, default='spiralMPC_empc_cost.db', help='Name of the database')
    parser.add_argument('--just_test_case', action='store_true', help='Calculate test the case (F31,1), (F41,1)')

    parser.add_argument('--tuning_file', type=str, default='spiral_mpc_empc.yaml', help='Path to the tuning file')
    parser.add_argument('--robot_param_file', type=str, default='robot_parameters_gazebo.yaml', help='Path to the robot parameter file')

    args = parser.parse_args()

    if not(args.overwrite) and not query_yes_no("Creating new cost functions will delete the " + \
            "existing ones and may take several hours. Continue? (Tip: Copy existing db as " + \
            "described in README.md)"):
            exit()

    # Get parameters
    tuning_params = read_ros_parameter_file(args.tuning_file)
    system_params = read_ros_parameter_file(args.robot_param_file)

    if not(args.overwrite) and not query_yes_no(f"Using the following parameters: " +  
                        f"\n\t- Tuning: {args.tuning_file}. " +
                        f"\n\t- Robot params: {args.robot_param_file}. " +
                        "Continue?"):
        print("You can choose different parameters using cmd line arguments. Use --help for help.")
        exit()
    else:
        print("Using the following parameters: " + f"\n\t- Tuning: {args.tuning_file}. " + \
              f"\n\t- Robot params: {args.robot_param_file}. ")

    pprint.pprint(tuning_params)
    pprint.pprint(system_params)

    cost_handler = CostHandler(args.db_name)

    if args.just_test_case:
        model = FreeFlyerDynamicsFull(tuning_params["time_step"], system_params)
        model.add_actuator_fault([3,1], 1)
        model.add_actuator_fault([4,1], 1)
        calculate_cost_fcn(model, cost_handler, tuning_params["tuning"][tuning_params["param_set"]])
        exit()
