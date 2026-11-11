
# Initialize both robots to make them ready to play in iPython
#   + import math libraries to do the math
#   + set joint and line speeds and linear ones

import urx
#import math3d as m3d
import numpy as np
import threading

vJoints = 0.5
aJoints = 0.01

vLine = 0.4
aLine = 0.01

# Initialize Robot1
rob1 = urx.Robot("192.168.56.101")
# TCP must be defined using the original coordinate system of the robot flange
#   Robot1 has a curved instrument and therefore needs a correction of 7 mm in the TCP
#rob1.mox((-0.37626284, -1.71089287, -1.15919388, -1.85352342,  1.55074358, -0.37225355))
# Gripper (RG2) weight is 0.78 kg
rob1.set_payload(0.78, (0.0, 0.0, 0.1))
# Force of gravity must be defined according to the mounting,
# i.e. gravity works againts y-axis
rob1.set_gravity((0.0, + 9.81, 0.0))

rob1.movej((-0.37626284, -1.71089287, -1.15919388, -1.85352342,  1.55074358, -0.37225355))
# # Initialize Robot2
# rob2 = urx.Robot("141.64.75.56")
# # TCP must be defined using the original coordinate system of the robot flange
# # Gripper (RG2 new) is 2 cm longer than the last version
# # Tool has a length of 69 mm

# #rob2.set_tcp((0.0, 0.0, + 0.26 + 0.02 + 0.069 + 0.005, 0.0, 0.0, 0.0))
# rob2.set_tcp((+ 0.00126, - 0.00253, + 0.33689, - 0.0043, - 0.0282, + 0.0301))
# # Gripper (RG2 new) weight is 0.79 kg
# rob2.set_payload(0.79, (0.0, 0.0, 0.1))
# # Force of gravity must be defined according to the mounting,
# # i.e. gravity works againts y-axis
# rob2.set_gravity((0.0, + 9.81, 0.0))


# print('WARNING: Robots are active\n')

# # print('Moving both robots to the home position ... out of singularities')
# # print('   Robot1: movej')
# # #rob1.movej(HomePositionJoints1, aJoints, vJoints)
# # print('   Robot2: movej\n')
# # #rob2.movej(HomePositionJoints2, aJoints, vJoints)

