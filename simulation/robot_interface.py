from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple


class RobotInterface(ABC):

    @abstractmethod
    def get_actuators_names(self):
        """
        Returns the list of actuators names in the robot.
        """
        pass

    @abstractmethod
    def get_joint_names(self):
        """
        Returns the list of joint names in the robot.
        """
        pass

    @abstractmethod
    def get_joints_telemetry(self, joint_names: Union[List[str], None] = None) -> Union[Tuple[Dict, Dict],None]:
        """
        Returns the current joint telemetry of the robot.
        """
        pass

    @abstractmethod
    def send_offline_joints_trajectory(self, joint_names: List[str], trajectory: List[float], time_reference: List[float]):
        """
        Offline send entire trajectory q[t0], ... , q[tk] to the robot.
        """
        pass

    @abstractmethod
    def send_joints_velocity_trajectory(self):
        """
        Offline send entire trajectory q_dot[t0], ... , q_dot[tk] to the robot.
        """
        pass

    @abstractmethod
    def send_joints_position_command(self, position_dict: Dict[str, float]):
        """
        Send q[tk] to the robot.
        """
        pass