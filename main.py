from flight_environment import FlightEnvironment
from path_planner import RRT3DStar, AStar3D
from trajectory_generator import TrajectoryGenerator

env = FlightEnvironment(100) #50
start = (1,2,0)
goal = (18,18,3)

# --------------------------------------------------------------------------------------------------- #
# Call your path planning algorithm here. 
# The planner should return a collision-free path and store it in the variable `path`. 
# `path` must be an N×3 numpy array, where:
#   - column 1 contains the x-coordinates of all path points
#   - column 2 contains the y-coordinates of all path points
#   - column 3 contains the z-coordinates of all path points
# This `path` array will be provided to the `env` object for visualization.

# rrtstar = RRT3DStar(env, start, goal, max_iter=10000, extend_length=0.05)
# path = rrtstar.planning()

planner = AStar3D(env, voxel_size=0.5)
path = planner.search(start, goal)

# print("path",path," len",len(path))
# path = [[1, 2, 0], [2.4677935445576997, 1.7941422698448999, 1.4162290441486272], [1.8288821309099412, 2.5959495185190726, 3.1914465917985413], [1.4116457062945953, 3.7287058425423214, 3.373815065561532], [3.06518799840564, 4.854053571674452, 2.924492935306069], [4.670718356282364, 6.128342194618639, 2.8934965585622616], [4.895624821250256, 8.120530897623674, 3.3213998813289383], [6.553613061057961, 9.320428449962328, 3.203837725211193], [7.170441031691286, 11.261807271733467, 3.4342103039242025], [8.668379324325558, 12.617876944526579, 3.0881526295537225], [9.510047708717025, 14.36045973310037, 3.7645397640279405], [11.21515532768441, 14.111850368513986, 2.653998728505467], [12.814559455222586, 15.3526738285376, 2.3303267064385684], [14.631059067870662, 14.797273936895413, 3.1012742517444973], [15.637845319629905, 16.31686326507086, 2.163335172185562], [17.296908959371887, 16.191015756954197, 0.9657667006621529], [17.809555658604083, 17.5100054006453, 2.4489928066898043], [18, 18, 3]]

# --------------------------------------------------------------------------------------------------- #


env.plot_cylinders(path)


# --------------------------------------------------------------------------------------------------- #
#   Call your trajectory planning algorithm here. The algorithm should
#   generate a smooth trajectory that passes through all the previously
#   planned path points.
#
#   After generating the trajectory, plot it in a new figure.
#   The figure should contain three subplots showing the time histories of
#   x, y, and z respectively, where the horizontal axis represents time (in seconds).
#
#   Additionally, you must also plot the previously planned discrete path
#   points on the same figure to clearly show how the continuous trajectory
#   follows these path points.

traj = TrajectoryGenerator(path)
traj.plot()

# --------------------------------------------------------------------------------------------------- #



# You must manage this entire project using Git. 
# When submitting your assignment, upload the project to a code-hosting platform 
# such as GitHub or GitLab. The repository must be accessible and directly cloneable. 
#
# After cloning, running `python3 main.py` in the project root directory 
# should successfully execute your program and display:
#   1) the 3D path visualization, and
#   2) the trajectory plot.
#
# You must also include the link to your GitHub/GitLab repository in your written report.
