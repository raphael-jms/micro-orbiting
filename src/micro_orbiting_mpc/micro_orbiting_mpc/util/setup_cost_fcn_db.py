import argparse
import pprint
import time
from datetime import timedelta
import yaml
import pprint

from micro_orbiting_mpc.controllers.spiralMPC_eMPC.explicit_mpc_terminal_incredients import explicitMPCTerminalIngredients
from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull
from micro_orbiting_mpc.util.cost_handler import CostHandler
from micro_orbiting_mpc.util.yes_no_question import query_yes_no
from micro_orbiting_mpc.util.utils import read_ros_parameter_file, frame_message

def calculate_cost_fcn(model, cost_handler, tuning, system_params):
    # Create the terminal cost function
    terminal_cost = explicitMPCTerminalIngredients(model, tuning)

    terminal_cost.calculate_terminal_ingredients()

    tcost = terminal_cost.get_terminal_cost()
    tset = terminal_cost.get_terminal_set()

    # Save the terminal cost function to the database
    cost_handler.set_cost_fcn(tcost, tset, model, tuning, system_params)

def get_tuning_parameters(tuning_params):
    if "tuning" in tuning_params.keys() and "param_set" in tuning_params.keys():
        # File structure for single controller
        return tuning_params["tuning"][tuning_params["param_set"]]
    elif "tuning" in tuning_params.keys() and "spiraling" in tuning_params["tuning"].keys():
        # File structure for reactive controller
        tuning_params = tuning_params["tuning"]["spiraling"]
        return tuning_params[tuning_params["param_set"]] 
    else:
        raise ValueError("I do not know this file structure") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters for setting up the term cost database")

    parser.add_argument('--overwrite', action='store_true', help='Overwrite the existing database')
    parser.add_argument('--yes', action='store_true', help='Confirm all questions')
    parser.add_argument('--resolution', type=int, default=4, help='Number of cases considered per thruster')
    parser.add_argument('--db_name', type=str, default='spiralMPC_empc_cost.db', help='Name of the database')
    parser.add_argument('--just_param_file_case', action='store_true', help='Calculate only the case specified in the parameter file')
    parser.add_argument('--cases', type=str, help='Cases to be calculated, takes yaml file as input of form list(list("act_ids": [int, int], "intensity": float))')

    parser.add_argument('--tuning_file', type=str, default='spiral_mpc_empc.yaml', help='Path to the tuning file')
    parser.add_argument('--robot_param_file', type=str, default='robot_parameters_gazebo.yaml', help='Path to the robot parameter file')

    args = parser.parse_args()

    # Ensure user wants to recalculate the cost functions
    if not(args.yes) and not query_yes_no("Creating new cost functions will delete the " + \
            "existing ones and may take several hours. Continue? \n(Tip 1: Copy existing db as " + \
            "described in README.md) \n(Tip 2: Skip this question with the --yes flag)\n"):
            exit()

    # Get parameters
    tuning_params = read_ros_parameter_file(args.tuning_file)
    system_params = read_ros_parameter_file(args.robot_param_file)

    # Ensure user uses the correct parameters
    if not(args.yes) and not query_yes_no(f"Using the following parameters: " +  
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

    # Create the cost handler
    cost_handler = CostHandler(args.db_name)

    # Delete and recreate the database newly if wanted
    if args.overwrite:
        cost_handler.create_table(overwrite=True)

    # Create the cost functions

    # If the flag is set, calculate only the case specified in the parameter file
    if args.just_param_file_case:
        start_t = time.time()

        model = FreeFlyerDynamicsFull(tuning_params["time_step"], system_params)
        for act in tuning_params["actuator_failures"]:
            model.add_actuator_fault(act['act_ids'], act['intensity'])
        calculate_cost_fcn(model, cost_handler, get_tuning_parameters(tuning_params),
                           system_params)

        elapsed = timedelta(seconds=time.time() - start_t)
        print("*"*41 + f"\n\n\tTime taken: {str(elapsed).split('.')[0]} h:m:s\n\n" + "*"*41)

        exit()
 
    # If the cases are specified in a file, calculate them
    if args.cases is not None:
        """
        Takes a YAML file as input of the form

        # Different cases to be calculated
        - 
            # Different actuators in each case
            - act_ids: [3,1]
            intensity: 1
            - act_ids: [4,1]
            intensity: 1
        - 
            - act_ids: [1,1]
            intensity: 1
            - act_ids: [2,1]
            intensity: 1

        Calculates the terminal ingredients for all cases
        """
        start_t = time.time()

        # read the cases from the file
        with open(args.cases, 'r') as file:
            failure_cases = yaml.safe_load(file)

        if not args.yes:
            print(f"Calculating the following cases:")
            for case in failure_cases:
                pprint.pprint(case)
            if not query_yes_no("Is this correct?"):
                print("You can choose different cases using the --cases argument. Use the following file format:\n")
                print('''        # Different cases to be calculated
        - 
            # Different actuators in each case
            - act_ids: [3,1]
            intensity: 1
            - act_ids: [4,1]
            intensity: 1
        - 
            - act_ids: [1,1]
            intensity: 1
            - act_ids: [2,1]
            intensity: 1)''')
                exit()

        for current_case_number, case in enumerate(failure_cases):
            model = FreeFlyerDynamicsFull(tuning_params["time_step"], system_params)

            for act in case:
                model.add_actuator_fault(act['act_ids'], act['intensity'])
            calculate_cost_fcn(model, cost_handler, get_tuning_parameters(tuning_params),
                               system_params)

            print(frame_message(f"Finished case {current_case_number+1} of {len(failure_cases)}", indent=4, add_newline=False))

        elapsed = timedelta(seconds=time.time() - start_t)
        print(frame_message(f"Time taken: {str(elapsed).split('.')[0]} h:m:s"))
        exit()

    # Calculate all the cases
    raise NotImplementedError("This script is not implemented fully yet. Please use the --just_param_file_case flag.")
    # TODO: Iterate over all combinations of actuator failures and calculate the cost functions
