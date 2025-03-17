import numpy as np
import bezier
import matplotlib.pyplot as plt
import random

# import seaborn
# seaborn.set()

def create_cubic_bezier_path(start_position: list, goal_position: list):
    """
    Creates a cubic Bezier curve with random intermediate points within a rectangle
    formed by the start and goal positions.

    Args:
        start_position (list): The [x, y] coordinates of the starting position.
        goal_position (list): The [x, y] coordinates of the goal position.

    Returns:
        bezier.Curve: The generated cubic Bezier curve.
    """
    # Generate random intermediate points within the rectangle
    intermediate_point_1 = [
        random.uniform(min(start_position[0], goal_position[0]), max(start_position[0], goal_position[0])),
        random.uniform(min(start_position[1], goal_position[1]), max(start_position[1], goal_position[1])),
    ]
    # intermediate_point_2 = [
    #     random.uniform(min(start_position[0], goal_position[0]), max(start_position[0], goal_position[0])),
    #     random.uniform(min(start_position[1], goal_position[1]), max(start_position[1], goal_position[1])),
    # ]
    intermediate_point_2 = [0,0]
    neg_start_position = [-i for i in start_position]
    intermediate_point_2[0] = neg_start_position[0] + goal_position[0]
    intermediate_point_2[1] = neg_start_position[1] + goal_position[1]
    # Create the nodes for the cubic Bezier curve
    nodes = np.asfortranarray([
        [start_position[0], intermediate_point_1[0], intermediate_point_2[0], goal_position[0]],
        [start_position[1], intermediate_point_1[1], intermediate_point_2[1], goal_position[1]],
    ])

    # Generate the Bezier curve
    curve = bezier.Curve.from_nodes(nodes)

    return curve


# Define the curves
# nodes1 = np.asfortranarray([
#     [0.0, 0.5, 1.0],
#     [0.0, 1.0, 0.0],
# ])
# quadratic_curve = bezier.Curve(nodes1, degree=2)

# nodes2 = np.asfortranarray([
#     [0.0, 0.25, 0.5, 0.75, 1.0],
#     [0.0, 2.0, -2.0, 2.0, 0.0],
# ])
# curve2 = bezier.Curve.from_nodes(nodes2)

# # Find intersections
# intersections = quadratic_curve.intersect(curve2)
# s_vals = np.asfortranarray(intersections[0, :])
# points = quadratic_curve.evaluate_multi(s_vals)

# # Plot
# ax = quadratic_curve.plot(num_pts=256)
# _ = curve2.plot(num_pts=256, ax=ax)
# lines = ax.plot(
#     points[0, :], points[1, :],
#     marker="o", linestyle="None", color="black"
# )


start = [0, 0]
goal = [5, 5]

# Generate a random cubic Bezier curve
curve = create_cubic_bezier_path(start, goal)

# Plot the curve
curve_points = curve.evaluate_multi(np.linspace(0, 1, 256))
plt.plot(curve_points[0, :], curve_points[1, :], label="Bezier Curve")

# Plot the control points
plt.scatter([start[0], goal[0]], [start[1], goal[1]], color="blue", label="Start/Goal")
plt.scatter(curve.nodes[0, 1:3], curve.nodes[1, 1:3], color="red", label="Control Points")

# Add labels and legend
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Random Cubic Bezier Curve")
plt.legend()
plt.grid(True)
plt.show()



