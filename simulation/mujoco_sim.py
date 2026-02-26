import os
import queue
import time
import mujoco
import mujoco.viewer
import numpy as np

from collections import deque
from multiprocessing import Process, Lock
from queue import Empty
from controller_push_IKQP.commons.logger import logger


class SimulationError(Exception):
    """Custom exception for simulation errors."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class MujocoWorker(Process):

    def __init__(self):
        super().__init__()
        self._renderer = None
        self._vis_deque = deque(maxlen=50)  # Store last 20 visualization points
        self._vis_lock = Lock()
        self._sim_is_running = False
        self.input_queue, self.output_queue, self.vis_queue, self.frames_queue = None, None, None, None
        self.model, self.data = None, None
        self.actuators_name_to_index_mapping, self.joint_names_to_index_mapping, self.joint_to_actuator_mapping = None, None, None
        self.joint_names , self.actuator_names = None, None
        self.actuator_idx_arr , self.joint_idx_arr = None, None

        self.req_queue = None
        self.resp_queue = None

        self.joint_name_to_qposadr = {}
        self.joint_name_to_dofadr = {}

        self.ee_geom_ids = set()
        self.box_geom_id = -1
        self.ee_body_id = -1   # optional, only if you also want body wrench sanity-check


    def setup_simulation(self, xml_path, input_queue, output_queue, vis_queue, frames_queue,req_queue, resp_queue, ee_geom_names):
        """ Initializes the MuJoCo simulation worker. Returns the list of joint names if configured successfully.
        :param xml_path: Path to the MuJoCo XML model file.
        :param input_queue: Queue for receiving commands (e.g., joint positions).
        :param output_queue: Queue for sending simulation outputs (e.g., joint positions).
        :param vis_queue: Queue for trajectory visualization.
        :param frames_queue: Queue for frames streaming.
        """

        self.req_queue = req_queue
        self.resp_queue = resp_queue

        if not os.path.exists(xml_path):
            logger.error(f"Error: XML file not found at {xml_path}")
            raise SimulationError(f"XML file not found at {xml_path}")

        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            logger.error(f"Error loading MuJoCo model: {e}")
            raise SimulationError(f"Failed to load MuJoCo model from {xml_path}")

        self.state = []
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.vis_queue = vis_queue
        self.frames_queue = frames_queue

        # Actuators mapping
        self.actuators_name_to_index_mapping = {}
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.actuators_name_to_index_mapping[actuator_name] = i

        # Joints mapping
        self.joint_names_to_index_mapping = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            self.joint_names_to_index_mapping[joint_name] = i

        # Joint-to-actuator mapping
        self.joint_to_actuator_mapping = {}
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]  # Get joint ID that actuator controls
            if joint_id >= 0:  # Valid joint
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_id)
                self.joint_to_actuator_mapping[joint_name] = i

        self.joint_name_to_qposadr = {}
        self.joint_name_to_dofadr = {}
        for jname, jid in self.joint_names_to_index_mapping.items():
            self.joint_name_to_qposadr[jname] = int(self.model.jnt_qposadr[jid])
            self.joint_name_to_dofadr[jname]  = int(self.model.jnt_dofadr[jid])


        # Saving names lists
        # Create fixed ordered lists of names and their corresponding MuJoCo indices
        self.actuator_names = list(self.actuators_name_to_index_mapping.keys())
        self.joint_names = list(self.joint_names_to_index_mapping.keys())

        # Pre-map which control index belongs to which actuator name
        self.actuator_idx_arr = np.arange(self.model.nu)
        # Pre-map which qpos index belongs to which joint name
        self.joint_idx_arr = np.arange(self.model.nq)

        # --- Contact filtering: EE geoms vs box geom ---
        # Update these names to whatever your XML uses for collision geoms

        self.ee_geom_ids = set()
        for gname in ee_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            if gid != -1:
                self.ee_geom_ids.add(gid)

        # Box geom name must match your scene XML. If you used name="box_geom", keep this:
        self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
        if self.box_geom_id == -1:
            # fallback if you still named it "box"
            self.box_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")

        # Optional: EE body id for sanity-check using cfrc_ext
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")  # change if needed


        return self.actuator_names , self.joint_names

    def _add_trajectory_visualization(self, viewer, waypoint):
        """Internal method to update the visualization"""
        # Append the point to the deque
        self._vis_deque.append(waypoint)
        l = len(self._vis_deque)
        if l < 3:
            return
        ngeom = 0  # Counter for used geoms
        logger.info(len(self._vis_deque))
        for i in range(len(self._vis_deque) - 1):
            from_pos = self._vis_deque[i]
            to_pos = self._vis_deque[i + 1]

            # ---- Sphere at from_pos ----
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.01, 0, 0]),  # Sphere radius
                pos=from_pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.2, 0.3, 0.3, 0.5])  # RGBA in [0,1]
            )
            ngeom += 1

            # ---- Line from from_pos to to_pos ----
            mujoco.mjv_connector(
                viewer.user_scn.geoms[ngeom],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=10,
                from_=from_pos,
                to=to_pos
            )
            ngeom += 1

        viewer.user_scn.ngeom = ngeom

    def push_telemetry(self, item):
        # 1. Drain the queue entirely to ensure we make space
        # and remove stale data.
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except (queue.Empty, ValueError):
                break

        # 2. Try to put the latest item.
        # If it still fails (rare race condition), we just skip this frame
        # to keep the simulation loop moving.
        try:
            self.output_queue.put_nowait(item)
        except queue.Full:
            # If the queue is still full, another process likely hasn't
            # processed the 'get' signal yet. We skip to avoid crashing.
            pass

    def run(self):
        if not self.model or not self.data:
            raise SimulationError("Simulation not initialized. Call setup_simulation first.")

        logger.info("Starting simulation...")

        # Initialize Renderer (Optional: handle context within the loop or here)
        self._renderer = mujoco.Renderer(self.model, 480, 640)

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Synchronize real-time clock with simulation clock
            start_time = time.time()

            old_cmd = None
            while self._sim_is_running and viewer.is_running():
                cmd = None
                try:
                    cmd = self.input_queue.get_nowait()
                    old_cmd = cmd
                except Empty:
                    cmd = old_cmd  # reuse last command

                # Apply cmd if available
                if cmd:
                    if 'actuators_control' in cmd:
                        for actuator_or_joint_name, val in cmd['actuators_control'].items():
                            # IMPORTANT: actuator mapping should be by actuator name, not joint name
                            if actuator_or_joint_name in self.actuators_name_to_index_mapping:
                                self.data.ctrl[self.actuators_name_to_index_mapping[actuator_or_joint_name]] = val
                            # elif actuator_or_joint_name in self.joint_names_to_index_mapping:
                            #     self.data.qpos[self.joint_names_to_index_mapping[actuator_or_joint_name]] = val

                    if 'joints_control' in cmd:
                        for joint_name, val in cmd['joints_control'].items():
                            if joint_name in self.joint_names_to_index_mapping:
                                self.data.qpos[self.joint_names_to_index_mapping[joint_name]] = val

                # Always step
                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                # Triggering Contact
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)

                # Push telemetry
                state = (
                    dict(zip(self.joint_names_to_index_mapping.keys(), self.data.qpos.copy())),
                    dict(zip(self.joint_names_to_index_mapping.keys(), self.data.qvel.copy()))
                )
                self.push_telemetry(state)

                # Handle kinematics requests (see next section)
                self._handle_requests()

    def _handle_requests(self):
        if self.req_queue is None or self.resp_queue is None:
            return

        while True:
            try:
                req = self.req_queue.get_nowait()
            except Empty:
                break

            rtype = req.get("type", "")
            frame = req.get("frame", "")

            if rtype == "pose":
                pose = self._compute_pose(frame)
                self._respond(req, pose)

            elif rtype == "jacobian":
                J = self._compute_jacobian(frame)
                self._respond(req, J)

            elif rtype == "pose_jacobian":
                pose = self._compute_pose(frame)
                J = self._compute_jacobian(frame)
                self._respond(req, (pose, J))

            elif rtype == "telemetry":
                joint_names = req.get("joint_names", None)
                payload = self._compute_telemetry(joint_names)
                self._respond(req, payload)
            elif rtype == "contacts":
                payload = self._compute_ee_box_contacts()
                self._respond(req, payload)


    def _respond(self, req, payload):
        msg = {
            "id": req.get("id", None),
            "type": req.get("type", ""),
            "frame": req.get("frame", ""),
            "payload": payload
        }
        # keep only latest response to avoid backlog
        while not self.resp_queue.empty():
            try:
                self.resp_queue.get_nowait()
            except Exception:
                break
        try:
            self.resp_queue.put_nowait(msg)
        except Exception:
            pass

    def sim_stop(self):
        self._sim_is_running = False

    def sim_kill(self):
        self._sim_is_running = False
        self.join()

    def sim_start(self):
        self._sim_is_running = True
        self.start()

    def get_joint_positions(self):
        state = None
        # Keep getting until the queue is empty to get the ABSOLUTE LATEST
        while True:
            try:
                state = self.output_queue.get_nowait()
            except Empty:
                break
        return state  # This is now the freshest possible data

    def _compute_pose(self, frame_name: str):
        mujoco.mj_kinematics(self.model, self.data)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if body_id == -1:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
            if site_id ==-1:
                return None
            else:
                return(self.data.site_xpos[site_id].copy(),
                       self.data.site_xmat[site_id].copy())
        return (self.data.xpos[body_id].copy(),
                self.data.xquat[body_id].copy())

    def _compute_jacobian(self, frame_name: str):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
        if body_id == -1:
            return None

        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

        nv = self.model.nv
        jac_p = np.zeros((3, nv), dtype=np.float64)
        jac_r = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, self.data, jac_p, jac_r, body_id)
        return np.vstack([jac_p, jac_r])

    def _compute_telemetry(self, joint_names=None):
        """
        Returns payload: [q_list, qd_list]
        where each is ordered according to joint_names (if provided),
        otherwise uses self.joint_names order.

        IMPORTANT: qpos/qvel indexing by joint id is not generally correct in MuJoCo.
        For an MVP, assume hinge/slide joints where joint_id aligns with qpos index.
        If your model includes free joints, this needs proper address mapping.
        """
        # Ensure kinematics/dynamics are current enough (mj_step already called in loop)
        # No need to mj_step here. If you manually change qpos elsewhere, call mj_forward.

        if joint_names is None:
            joint_names = self.joint_names
        joint_names = list(joint_names)

        q_list = []
        qd_list = []

        for jn in joint_names:
            if jn not in self.joint_name_to_qposadr or jn not in self.joint_name_to_dofadr:
                q_list.append(float("nan"))
                qd_list.append(float("nan"))
                continue

            qadr = self.joint_name_to_qposadr[jn]
            dadr = self.joint_name_to_dofadr[jn]

            q_list.append(float(self.data.qpos[qadr]))
            qd_list.append(float(self.data.qvel[dadr]))

        return [q_list, qd_list]

    def _compute_ee_box_contacts(self):
        """
        Returns:
          {
            "ncon_total": int,
            "contacts": [
              {
                "pos": [x,y,z],
                "normal_world": [nx,ny,nz],
                "force_world": [fx,fy,fz],
                "torque_world": [tx,ty,tz],
                "force_contact": [fn,ft1,ft2],
                "torque_contact": [tn,tt1,tt2],
                "geom1": "name",
                "geom2": "name",
              }, ...
            ],
            "sum_force_world": [Fx,Fy,Fz],
            "sum_torque_world": [Tx,Ty,Tz],
          }
        """
        if self.box_geom_id == -1 or len(self.ee_geom_ids) == 0:
            return {
                "error": "Missing geom ids. Check box geom name and EE geom names.",
                "box_geom_id": int(self.box_geom_id),
                "ee_geom_ids": [int(x) for x in self.ee_geom_ids],
                "ncon_total": int(self.data.ncon),
                "contacts": [],
                "sum_force_world": [0.0, 0.0, 0.0],
                "sum_torque_world": [0.0, 0.0, 0.0],
            }

        contacts_out = []
        sum_f = np.zeros(3, dtype=np.float64)
        sum_tau = np.zeros(3, dtype=np.float64)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)

            is_ee_box = ((g1 in self.ee_geom_ids and g2 == self.box_geom_id) or
                         (g2 in self.ee_geom_ids and g1 == self.box_geom_id))
            if not is_ee_box:
                continue

            out6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, out6)

            # Contact frame in world coords (3x3). By convention, x-axis is the contact normal.
            R = c.frame.reshape(3, 3)

            f_world = R @ out6[0:3]
            tau_world = R @ out6[3:6]
            n_world = R[:, 0].copy()

            # Determine sign convention consistently:
            # mj_contactForce returns force on geom1 (in the contact frame). If you want "force on EE",
            # flip sign when EE is geom2.
            force_on_ee_world = f_world.copy()
            torque_on_ee_world = tau_world.copy()
            if g2 in self.ee_geom_ids:
                force_on_ee_world = -force_on_ee_world
                torque_on_ee_world = -torque_on_ee_world
                n_world = -n_world  # normal pointing toward EE for coherence

            sum_f += force_on_ee_world
            sum_tau += torque_on_ee_world

            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1) or f"geom{g1}"
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2) or f"geom{g2}"

            contacts_out.append({
                "pos": c.pos.copy().tolist(),
                "normal_world": n_world.tolist(),
                "force_world": force_on_ee_world.tolist(),
                "torque_world": torque_on_ee_world.tolist(),
                "force_contact": out6[0:3].copy().tolist(),
                "torque_contact": out6[3:6].copy().tolist(),
                "geom1": name1,
                "geom2": name2,
            })

        return {
            "ncon_total": int(self.data.ncon),
            "contacts": contacts_out,
            "sum_force_world": sum_f.tolist(),
            "sum_torque_world": sum_tau.tolist(),
        }
