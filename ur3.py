from urx.robot import Robot
from urx.gripper import OnRobotGripperRG2
from urx.gripper import OnRobotGripperRG20
import numpy as num
from urx import ursecmon
import time


class UR3(Robot):

    def __init__(self, host, robot_nr):
        for i in range(0, 11):
            try:
                super().__init__(host)
                time.sleep(.5)
                break
            except ursecmon.TimeoutException:
                if i < 10:
                    self.logger.warning("Couldn't connect to " + str(self.host) + " - " + str(i))
                    continue
                else:
                    self.logger.error("Couldn't connect to " + str(self.host) + " - " + str(i))
                    raise

        self.logger.debug("Connected to " + str(self.host))
        self.robot_nr = robot_nr

    # Move to home position
    def home(self, wait=True):
        self.movej((0, num.deg2rad(-90), 0, num.deg2rad(-90), 0, 0), 0.1, 0.3, wait=wait)

    # Display message in popup window in GUI
    def popup(self, s, title="Popup", warning=False, error=False, blocking=False):
        self.send_program(f'popup("{s}", title="{title}", warning={warning}, error={error}, blocking={blocking})')

    # Shutdown the robot, and power off the robot and controller
    def powerdown(self):
        self.send_program('powerdown()')


# Move joint position of to_rob to joint position of from_rob
def sync_position(from_rob: UR3, to_rob: UR3, acc=0.1, vel=0.05):
    to_rob.movej(from_rob.getj(), acc=acc, vel=vel)
