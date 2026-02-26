import time
import uuid
import numpy as np
from multiprocessing import Queue
from queue import Empty
from typing import Dict, List, Union, Tuple, Any, Optional

from hybrid_push_control_panda.simulation.robot_interface import RobotInterface
from hybrid_push_control_panda.simulation.mujoco_sim import SimulationError, MujocoWorker
from hybrid_push_control_panda.commons.logger import logger

class MujocoInterface(RobotInterface):

    def __init__(self, xml_path: str, geometry_names: Optional[List[str]] = None):
        """Initializes the MuJoCo interface with the specified XML model path."""
        self._input_queue = Queue()
        self._output_queue = Queue(maxsize=1)
        self._vis_queue = Queue()
        self._frames_queue = Queue()

        # --- NEW: request/response queues for pose/jacobian (Step 4) ---
        self._req_queue = Queue(maxsize=50)
        self._resp_queue = Queue(maxsize=5)

        self.mj_sim = None
        try:
            self.mj_sim = MujocoWorker()
            self._actuators_names, self._joint_names = self.mj_sim.setup_simulation(
                xml_path=xml_path,
                input_queue=self._input_queue,
                output_queue=self._output_queue,
                vis_queue=self._vis_queue,
                frames_queue=self._frames_queue,
                req_queue=self._req_queue,
                resp_queue=self._resp_queue,
                ee_geom_names= geometry_names,
            )
            self.mj_sim.sim_start()
            time.sleep(2)
            logger.info("MuJoCo worker started successfully.")

        except SimulationError as e:
            logger.info(f"Error starting MuJoCo worker: {e}")
            if self.mj_sim:
                self.mj_sim.sim_stop()
            logger.info("MuJoCo worker stopped.")

    def __del__(self):
        logger.info("Killing mujoco worker...")
        if self.mj_sim and self.mj_sim.is_alive():
            self.mj_sim.sim_stop()
            time.sleep(1)
            self.mj_sim.join(timeout=2)
            if self.mj_sim.is_alive():
                self.mj_sim.terminate()

    @classmethod
    def create(cls, xml_path: str):
        instance = cls(xml_path)
        return instance, instance.get_actuators_names(), instance.get_joint_names()

    def get_actuators_names(self):
        return self._actuators_names

    def get_joint_names(self):
        return self._joint_names

    # ---------------- telemetry ----------------

    def get_joints_telemetry(self,
                             joint_names: Union[List[str], None] = None,
                             timeout: float = 0.2) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        """
        RPC-based telemetry fetch from worker.

        Returns:
          (q, qd) where:
            q  is (n,) np.ndarray joint positions
            qd is (n,) np.ndarray joint velocities
        Ordering:
          - If joint_names is provided, returned vectors follow that order.
          - Else, they follow self.get_joint_names() order.

        Requires worker to implement request type "telemetry".
        """
        if joint_names is None:
            joint_names = self.get_joint_names()
        joint_names = list(joint_names)

        payload = self._rpc("telemetry", frame_name="", timeout=timeout, extra={"joint_names": joint_names})
        if payload is None:
            return None

        q_list, qd_list = payload  # worker sends lists (picklable)
        q = np.asarray(q_list, dtype=float).reshape(-1)
        qd = np.asarray(qd_list, dtype=float).reshape(-1)
        return q, qd

    def get_joints_vel(self, joint_names: Union[List[str], None] = None) -> Union[Tuple[Dict, Dict], None]:
        return self.mj_sim.get_joint_positions()

    # ---------------- commands ----------------

    def send_offline_joints_trajectory(self, joint_names: List[str], trajectory: List[List[float]], time_reference: List[float]):
        assert len(trajectory) == len(time_reference), "Trajectory and time reference must have the same length"
        start_time = time.time()
        for i, q in enumerate(trajectory):
            target_time = start_time + time_reference[i]
            sleep_time = target_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._input_queue.put({'joints_control': {name: q[j] for j, name in enumerate(joint_names)}})

    def set_initial_joints_configuration(self, initial_joints_config):
        self._input_queue.put({'joints_control': initial_joints_config})

    def send_joints_velocity_trajectory(self):
        pass

    def send_joints_position_command(self, position_dict: Dict[str, float]):
        self._input_queue.put({'joints_control': position_dict})

    def set_control(self, control_dict: Dict[str, float]):
        self._input_queue.put({'actuators_control': control_dict})

    # ---------------- Step 4: request/response for pose/jacobian ----------------

    def _rpc(self, req_type: str, frame_name: str, timeout: float = 0.2, extra: dict | None = None):
        """
        Generic request/response call to the worker process.
        req_type: "pose", "jacobian", or "pose_jacobian"
        """
        req_id = str(uuid.uuid4())
        req = {"id": req_id, "type": req_type, "frame": frame_name, "t": time.time()}
        if extra:
            req.update(extra)

        # Put request (best-effort)
        try:
            self._req_queue.put_nowait(req)
        except Exception:
            # If the queue is full, drop oldest requests by draining once, then retry.
            try:
                while True:
                    self._req_queue.get_nowait()
            except Empty:
                pass
            self._req_queue.put(req)

        # Wait for matching response
        t0 = time.time()
        stash = []
        while (time.time() - t0) < timeout:
            try:
                msg = self._resp_queue.get_nowait()
            except Empty:
                time.sleep(0.001)
                continue

            if isinstance(msg, dict) and msg.get("id") == req_id:
                # Put back unrelated messages
                for other in stash:
                    try:
                        self._resp_queue.put_nowait(other)
                    except Exception:
                        pass
                return msg.get("payload")

            stash.append(msg)

        # Put back what we consumed but didn't use
        for other in stash:
            try:
                self._resp_queue.put_nowait(other)
            except Exception:
                pass

        return None

    def get_pose(self, frame_name: str):
        """
        Returns: (pos(3,), quat(4,)) from worker, or None on timeout.
        """
        return self._rpc("pose", frame_name, timeout=0.2)

    def get_jacobian(self, frame_name: str):
        """
        Returns: J (6 x nv) from worker, or None on timeout.
        """
        return self._rpc("jacobian", frame_name, timeout=0.2)

    def get_pose_and_jacobian(self, frame_name: str):
        """
        Optional convenience method to reduce IPC: get both in one request.
        Returns: ((pos, quat), J) or None.
        """
        return self._rpc("pose_jacobian", frame_name, timeout=0.25)

    def get_contact_forces(self, timeout: float = 0.05):
        """
        Returns payload from worker _compute_ee_box_contacts() only of the specified frames:
          {
            "ncon_total": int,
            "contacts": [...],
            "sum_force_world": [Fx,Fy,Fz],
            "sum_torque_world": [Tx,Ty,Tz],
          }
        """
        return self._rpc("contacts", frame_name="", timeout=timeout)

    # ---------------- unimplemented placeholders ----------------

    def get_torques(self, actuator_names):
        pass

    def get_obj_position(self, obj_name):
        pass

    def set_obj_position(self, obj_name, position):
        pass

    def visualize_trajectory(self, cartesian_position: np.ndarray):
        self._vis_queue.put(cartesian_position)

    def close(self):
        self.__del__()
