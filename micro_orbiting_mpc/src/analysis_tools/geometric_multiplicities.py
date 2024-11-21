import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from scipy.spatial import ConvexHull

"""
Check and plot the geometric multiplicities of occurring error cases. Multiplicities arise when
in one scenario actuators fail and their relative position is equal to the failures of another 
scenario. This means the scenarios are just rotated or mirrored. 

Basically, this code finds the multiplicities by checking 2D polytopes for similar shapes.

Usage: Change the variable no_failures. The code will calculate all scenarios with that amount 
of actuator failures and count and plot the equivalent scenarios.
The code is only intended to work for 3 and more failures.
"""


def angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    return np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    
def dist(p1, p2):
    return np.linalg.norm(p1-p2, 2)

def shape_signature(vertices):
    # Convert to numpy array if it's not already
    vertices = np.array(vertices)
    
    # Calculate center point
    center = np.mean(vertices, axis=0)
    
    # Calculate angles from center to each point
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    
    # Sort vertices by angle
    sorted_vertices = vertices[np.argsort(angles)]
    
    # Calculate angles between adjacent points
    angle_list = [angle(sorted_vertices[i-1], sorted_vertices[i], sorted_vertices[(i+1)%len(sorted_vertices)]) 
                  for i in range(len(sorted_vertices))]
    length_list = [dist(sorted_vertices[i-1], sorted_vertices[i]) for i in range(len(sorted_vertices))]
    
    # Normalize angles to be between 0 and 2pi
    angle_list = [(a + 2*np.pi) % (2*np.pi) for a in angle_list]
    
    # Find all rotations of the angle list
    rotations_angle = [angle_list[i:] + angle_list[:i] for i in range(len(angle_list))]
    rotations_length = [length_list[i:] + length_list[:i] for i in range(len(length_list))]
    
    # Also consider the reversed list for reflection symmetry
    reverse_rotations_angle = [angle_list[::-1][i:] + angle_list[::-1][:i] for i in range(len(angle_list))]
    reverse_rotations_length = [length_list[::-1][i:] + length_list[::-1][:i] for i in range(len(length_list))]
    
    # Combine original and reversed rotations
    all_rotations_angle = rotations_angle + reverse_rotations_angle
    all_rotations_length = rotations_length + reverse_rotations_length
    
    # Return the rotation that gives the "smallest" angle list lexicographically
    print(tuple(min(all_rotations_angle)))
    print(tuple(min(all_rotations_length)))
    return tuple([tuple(min(all_rotations_angle)), tuple(min(all_rotations_length))])

points = np.array([
    [2,1], [1,2], [-1,2], [-2,1],
    [-2,-1], [-1,-2], [1,-2], [2,-1]
])

def plot_thrusters():
    plt.scatter(points[:,0], points[:,1])

no_failures = 3  # Change this to test different numbers of failing thrusters
failing_input_combinations = list(combinations(points, no_failures))

result = defaultdict(list)
for failing_combination in failing_input_combinations:
    signature = shape_signature(failing_combination)
    print(signature)
    result[signature].append(failing_combination)


# Plotting
for i, (signature, equivalent_combinations) in enumerate(result.items()):
    fig = plt.figure(i+1)
    for j, combination in enumerate(equivalent_combinations):
        ax = fig.add_subplot(1, len(equivalent_combinations), j+1)
        vertices = np.array(combination)
        hull = ConvexHull(vertices)
        for simplex in hull.simplices:
            plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-')
        ax.axis('equal')
        ax.set_xlim(-2.2, 2.2)
        plot_thrusters()

plt.show()

print("Number of different formations:", len(result))
