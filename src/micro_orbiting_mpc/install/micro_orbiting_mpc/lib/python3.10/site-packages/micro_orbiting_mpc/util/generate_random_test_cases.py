import random
import pprint
import yaml
import os
from micro_orbiting_mpc.models.ff_dynamics import FreeFlyerDynamicsFull

if __name__ == "__main__":
    model = FreeFlyerDynamicsFull(0.1)

    # Generate random test cases
    cases = []

    """
    Create each 4 failure cases for 2 and 3 failures
    """

    for i in range(2):
        for des_number_of_failures in [2, 3]:
            current_case = []
            current_case_dict = []
            while len(current_case) < des_number_of_failures:
                randN = random.randint(1, 7)
                if not randN in current_case:
                    current_case.append(randN)
                    current_case_dict.append({
                        "act_ids": model.act_idx2pos(randN),
                        "intensity": 1
                    })

            cases.append(current_case_dict)

    pprint.pprint(cases)
    with open(os.path.expanduser("~/Documents/test_cases.yaml"), "w") as file:
        yaml.dump(cases, file)