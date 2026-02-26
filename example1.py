"""
Project: Controlled Push IKQP
Author: Gonzalo Meza (https://sites.google.com/view/gonzalomeza)
Date: January 2026
Description:
    Main execution script for the Franka Emika Panda push task.
    Implements a Finite State Machine (FSM) to handle approach,
    contact detection, and hybrid force/position control.
License: No License
"""
import os
import time
import numpy as np
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R

# Custom Imports
from simulation.mujoco_interface import MujocoInterface
from commons.logger import logger
from controllers.ik_qp import InverseKinematicsController, AdmittanceFilter
from controllers.ik_qp import LowPassFilter, AdmittanceFilterParams

# ==================================================================================================
# 1. CONFIGURATION & CONSTANTS
# ==================================================================================================

# --- File Paths ---
BASE_DIR = os.path.expanduser("~/workspace")  # <=============================== WRITE HERE YOUR BASE DIRECTORY
XML_FILE = os.path.join(BASE_DIR, "controller_push_IKQP", "model", "scene.xml")

# --- Geometry & Actuator Definition ---
LEFT_EE_GEOMS = ["finger_l1", "finger_l2", "finger_l3", "finger_l4", "finger_l5"]
RIGHT_EE_GEOMS = ["finger_r1", "finger_r2", "finger_r3", "finger_r4", "finger_r5"]
EE_COVER_GEOMS = ["hand_cover0", "hand_cover1", "hand_cover2", "hand_cover3", "hand_cover4", "hand_cover5"]
OBJECT_NAMES = ["box_geom"]

# Sets for efficient contact checking
LEFT_SET = set(LEFT_EE_GEOMS)
RIGHT_SET = set(RIGHT_EE_GEOMS)
EE_SET = set(EE_COVER_GEOMS)
ITEM_SET = set(OBJECT_NAMES)

# --- Control Parameters ---
N_DOFS = 7  # All controlled degrees of freedom
DT_CTRL = 0.01  # Control timestep (100Hz)
FRAME_CONTROL = "hand" # Control frame (end-effector) for IK and force control

# Force Regulation Targets
DELTA_X_TARGET  = 0.20    # <================= Desired displacement along X during force reg
DELTA_Y_TARGET  = -0.05   # <================= Desired displacement along Y during force reg
DELTA_TOL       = 3e-3  # Tolerance for reaching target (Control error + contact disturbances)
Z_FORCE         = 20.0  # Target force along Z during push phase (as logged originally)

# --- Initial Configuration ---
# Joint limits (radians)
Q_MIN = np.array([-np.pi, -1.7628, -np.pi, -3.0718, -np.pi, -0.0175, -np.pi])
Q_MAX = np.array([np.pi, 1.7628, np.pi, -0.0698, np.pi, 3.7525, np.pi])

# Velocity limits
DQ_MIN = -20.0 * np.ones(N_DOFS)
DQ_MAX = 20.0 * np.ones(N_DOFS)

# Initial Posture
q_arm_init = [0.0, 0, 0, -1.3, 0, 1.51, 0]
q_ee_init = [255]

# Initial IK weights
joint_vel_weight = 1e5
pos_weight, orientation_weight = 1, 1

# ==================================================================================================
# 2. FSM CLASSES
# ==================================================================================================
class State(Enum):
    IDLE = auto()
    APPROACH = auto()
    SEEK_CONTACT = auto()
    PRELOAD = auto()
    PUSH = auto()
    STOP = auto()
    RETRACT = auto()
    RECOVER_CONTACT = auto()
    DONE = auto()

class FsmContext:
    """ Context class to hold shared variables and interfaces across FSM states."""
    def __init__(self, F_push: float = 20.0):
        self.t_in_state = 0.0
        self.pose_ref = None
        self.quat_ref = None
        self.wrench_ref = (0,0,0,0,0,0)
        self.joint_vel_weight = 1e5

        # Original script state variables
        self.p_start_regulation = None
        self.curr_p = None
        self.curr_quat = None

        # Interfaces and controllers
        self.interface = None
        self.ik = None
        self.admittance = None
        self.lp_filter = None

        # Parameters for FSM logic
        self.params = type('Params', (), {})()
        self.params.F_preload = 10.0
        self.params.F_push = F_push
        self.params.seek_contact_timeout_s = 2.0
        self.params.recover_timeout_s = 2.0
        self.params.preload_timeout_s = 5.0
        self.params.push_timeout_s = 30.0

class FiniteStateMachine:
    """ Implements the FSM logic for the pushing task."""
    def __init__(self, ctx: FsmContext):
        self.fsm_ctx = ctx
        self.state = State.IDLE

    def set_state(self, new_state: State):
        self.on_exit(self.state)
        self.state = new_state
        self.fsm_ctx.t_in_state = 0.0
        self.on_enter(new_state)

    def on_enter(self, state: State):
        ctx = self.fsm_ctx
        if state == State.APPROACH:
            ctx.joint_vel_weight = 5e3
            ctx.wrench_ref = (0,0,0,0,0,0) #TODO CHeck if this is needed or if the impedance control can handle it by itself with zero wrench_ref
        elif state == State.SEEK_CONTACT:
            ctx.joint_vel_weight = 1e6
        elif state == State.PRELOAD:
            ctx.joint_vel_weight = 1e8
            ctx.wrench_ref = (0, 0, ctx.params.F_preload, 0, 0, 0)
            # Original logic from CLOSE_GRIPPER
            ctx.interface.set_control({"actuator8": 0})
        elif state == State.PUSH:
            ctx.joint_vel_weight = 1e8
            ctx.wrench_ref = (0, 0, ctx.params.F_push, 0, 0, 0)
            ctx.p_start_regulation = np.copy(ctx.curr_p)
        elif state == State.RETRACT:
            ctx.joint_vel_weight = 1e7
            ctx.wrench_ref = (0,0,0,0,0,0)
            # Original logic from OPEN_GRIPPER
            ctx.interface.set_control({"actuator8": 255})
            ctx.pose_ref = ctx.curr_p + np.array([0.0, 0.0, 0.1])
        elif state == State.STOP:
            ctx.wrench_ref = (0,0,0,0,0,0)

    def on_exit(self, state: State):
        pass

    def tick(self, dt: float):
        ctx = self.fsm_ctx
        ctx.t_in_state += dt

        if self.state == State.IDLE:
            return

        if self.state == State.APPROACH:
            # Pre-calculate target points based on object position
            obj_pose, _ = ctx.ik.extract_position_quat_from_pose(ctx.interface.get_pose("box"))
            ctx.pose_ref = obj_pose + np.array([0.0, 0.0, 0.2])  # 20cm above

            # -- Transition Condition [f1]: Reached Approach Position --
            pos_err = np.linalg.norm(ctx.pose_ref - ctx.curr_p)
            # Note: quat_cur is handled in main loop telemetry
            ori_err = ctx.ik.quat_error_eps_form(ctx.quat_ref, ctx.curr_quat)

            if pos_err < 3e-3 and all(ori_err < 5e-2):
                logger.info(f"[APPROACH] Position reached (Err: {pos_err:.4f}). Transition -> SEEK_CONTACT")
                self.set_state(State.SEEK_CONTACT)

        elif self.state == State.SEEK_CONTACT:
            obj_pose, _ = ctx.ik.extract_position_quat_from_pose(ctx.interface.get_pose("box"))
            ctx.pose_ref = obj_pose + np.array([0.0, 0.0, 0.11])  # Slightly above the surface because box is 0.1m high

            # -- Transition Condition [f2]: Contact Detection --
            contact_dict = ctx.interface.get_contact_forces()
            if len(contact_dict.get("contacts", [])) > 0:
                logger.info(f"[SEEK_CONTACT] Contact detected. Transition -> PRELOAD")
                ctx.p_start_regulation = np.copy(ctx.curr_p)
                self.set_state(State.PRELOAD)
            elif ctx.t_in_state > ctx.params.seek_contact_timeout_s:
                logger.warning(f"[SEEK_CONTACT] Contact not detected on time. Transition -> RECOVER CONTACT")
                self.set_state(State.RECOVER_CONTACT)

        elif self.state == State.RECOVER_CONTACT:
            ctx.pose_ref = ctx.pose_ref - np.array([0.0, 0.0, 1e-4])  # Small downward step to encourage contact if missed
            # -- Transition Condition [f2]: Contact Detection --
            contact_dict = ctx.interface.get_contact_forces()
            if len(contact_dict.get("contacts", [])) > 0:
                logger.info(f"[RECOVER CONTACT] Contact detected. Transition -> PRELOAD")
                ctx.p_start_regulation = np.copy(ctx.curr_p)
                self.set_state(State.PRELOAD)
            elif ctx.t_in_state > ctx.params.recover_timeout_s:
                logger.warning(f"[RECOVER CONTACT] Contact not detected after recovery attempts. Transition -> STOP")
                self.set_state(State.STOP)

        elif self.state == State.PRELOAD:
            # -- Transition Condition: Both fingers gripping box --
            contact_dict = ctx.interface.get_contact_forces()
            left_touched_geoms = set()
            right_touched_geoms = set()

            for contact in contact_dict.get("contacts", []):
                pair = {contact["geom1"], contact["geom2"]}
                if (pair & LEFT_SET) and (pair & ITEM_SET):
                    left_touched_geoms |= (pair & LEFT_SET)
                if (pair & RIGHT_SET) and (pair & ITEM_SET):
                    right_touched_geoms |= (pair & RIGHT_SET)

            if len(left_touched_geoms) >= 2 and len(right_touched_geoms) >= 2:
                logger.info(f"[PRELOAD] Stable Grip Confirmed. Transition -> PUSH")
                ctx.p_start_regulation = np.copy(ctx.curr_p)
                self.set_state(State.PUSH)
            elif ctx.t_in_state > ctx.params.preload_timeout_s:
                self.set_state(State.RECOVER_CONTACT)

        elif self.state == State.PUSH:
            # 1. Sense Force
            contact_dict = ctx.interface.get_contact_forces()
            measured_force_z = 0.0
            for contact in contact_dict.get("contacts", []):
                pair = {contact["geom1"], contact["geom2"]}
                if (pair & EE_SET) and (pair & ITEM_SET):
                    measured_force_z = contact['force_world'][2]
                    break

            # Filtering the force
            filtered_force = ctx.lp_filter.filter(x=measured_force_z, dt=dt)

            # 2. Admittance Control Step
            Fz_err = ctx.params.F_push - filtered_force
            dz_adj, v, z = ctx.admittance.step(f_ext=Fz_err, dt=dt)

            # 3. Update Target (Move X/Y, Regulate Z)
            ctx.pose_ref = ctx.p_start_regulation + np.array([DELTA_X_TARGET, DELTA_Y_TARGET, -dz_adj])

            # -- Transition Condition [f3]: Displacement Reached --
            dist_moved_x = ctx.curr_p[0] - ctx.p_start_regulation[0]
            dist_moved_y = ctx.curr_p[1] - ctx.p_start_regulation[1]

            x_reached = ( dist_moved_x >= DELTA_X_TARGET - DELTA_TOL if DELTA_X_TARGET >= 0 else dist_moved_x <= DELTA_X_TARGET + DELTA_TOL)
            y_reached = (dist_moved_y >= DELTA_Y_TARGET - DELTA_TOL if DELTA_Y_TARGET >= 0 else dist_moved_y <= DELTA_Y_TARGET + DELTA_TOL )

            logger.debug(f"[PUSH] dX: {dist_moved_x:.4f}/{DELTA_X_TARGET} | dY: {dist_moved_y:.4f}/{DELTA_Y_TARGET}| Regulating Z: {dz_adj:.4f}m"
                         f"| Controlled Force: {filtered_force:.2f}/{ctx.params.F_push}N")

            if x_reached and y_reached:
                logger.info(f"[PUSH] Target displacement reached. Transition -> RETRACT")
                self.set_state(State.RETRACT)
            elif ctx.t_in_state > ctx.params.push_timeout_s:
                logger.warning("[PUSH] Timeout reached without achieving target displacement. Transition -> STOP")
                self.set_state(State.STOP)

        elif self.state == State.STOP:
            # Similar to OPEN_GRIPPER transition start
            logger.error("[STOP] | Stopping motion and opening gripper. Transition -> RETRACT")
            self.set_state(State.RETRACT)

        elif self.state == State.RETRACT:
            # -- Transition Condition: Task End --
            if np.linalg.norm(ctx.pose_ref[2] - ctx.curr_p[2]) < 1e-2:
                logger.info(f"[RETRACT] Retraction complete. TASK COMPLETED.")
                self.set_state(State.DONE)

# ==================================================================================================
# 3. INITIALIZATION
# ==================================================================================================

# --- MuJoCo Interface Setup ---
geometry_contact_names = LEFT_EE_GEOMS + RIGHT_EE_GEOMS + EE_COVER_GEOMS
robot_interface = MujocoInterface(XML_FILE, geometry_contact_names)

actuator_names = robot_interface.get_actuators_names()
ee_name = actuator_names[-1]
robot_joint_names = robot_interface.get_joint_names()

logger.info(f"[INIT] | Joints: {len(robot_joint_names)} | Actuators: {len(actuator_names)} | EE: {ee_name}")

# --- Send Initial Hardware Command ---
position_command = {n: v for n, v in zip(robot_joint_names, np.zeros(len(actuator_names)))}
robot_interface.send_joints_position_command(position_command)

# --- Controller Setup ---
ikQP = InverseKinematicsController(robot_interface, n_dofs=N_DOFS)
params = AdmittanceFilterParams(m=2.0, d=200.0, k=1, max_x=0.3, max_v=0.2, max_a=2.0)

# --- 1. Capture Current State ---
p0, quat0 = ikQP.extract_position_quat_from_pose(robot_interface.get_pose(FRAME_CONTROL))
quat_cur_init = ikQP.change_quaternion_xyzw(quat0)

# --- 2. Define Orientation Targets (Euler to Quaternion) ---
phi, theta, psi = np.deg2rad(0), np.deg2rad(180), np.deg2rad(90)  # Z, Y, X
R_euler = R.from_euler('zyx', [phi, theta, psi]).as_matrix()

# Apply rotation (global frame)
quat_des_global = R.from_matrix(R_euler).as_quat()
quat_des_global = ikQP.change_quaternion_xyzw(quat_des_global)
quat_des_global = ikQP.check_quaternion(quat_des_global)

# --- 3. Set Initial Sim State ---
logger.info("[INIT] | Setting initial configuration and stabilizing...")
logger.info(f"[INIT] | Setting displacement targets for force regulation: X: {DELTA_X_TARGET:.2f}m | Y: {DELTA_Y_TARGET:.2f}m")
logger.info(f"[INIT] | Setting force regulation target Z-axis: {Z_FORCE} N") # Target Z force as logged originally
q_init_combined = np.array([q_arm_init + q_ee_init]).flatten()
actuator_command = {n: v for n, v in zip(actuator_names, q_init_combined)}
robot_interface.set_control(actuator_command)
time.sleep(2.0)  # Allow physics to settle

# --- Setup Context and FSM ---
fsm_ctx = FsmContext(F_push = Z_FORCE)
fsm_ctx.interface = robot_interface
fsm_ctx.ik = ikQP
fsm_ctx.admittance = AdmittanceFilter(params, x0=0.0, v0=0.0)
fsm_ctx.lp_filter = LowPassFilter(cutoff_hz=50)
fsm_ctx.quat_ref = quat_des_global

fsm = FiniteStateMachine(fsm_ctx)
fsm.set_state(State.APPROACH)

# ==================================================================================================
# 4. MAIN CONTROL LOOP
# ==================================================================================================

logger.info(f"[START] | Control loop started. Initial State: {fsm.state.name}")

while fsm.state != State.DONE:
    loop_start_time = time.time()

    # ----------------------------------------------------------------------------------------------
    # I. TELEMETRY UPDATE
    # ----------------------------------------------------------------------------------------------
    q_dict = robot_interface.get_joints_telemetry()
    q = q_dict[0].reshape(-1)[:N_DOFS]
    fsm_ctx.curr_p, quat_cur = ikQP.extract_position_quat_from_pose(robot_interface.get_pose(FRAME_CONTROL))
    quat_cur = ikQP.change_quaternion_xyzw(quat_cur)
    fsm_ctx.curr_quat = ikQP.check_quaternion(quat_cur)

    # ----------------------------------------------------------------------------------------------
    # II. STATE MACHINE LOGIC
    # ----------------------------------------------------------------------------------------------
    # Note: The FSM logic updates the pose_ref, quat_ref, wrench_ref, and joint_vel_weight in the context based on the current state and transitions.
    fsm.tick(DT_CTRL)

    # ----------------------------------------------------------------------------------------------
    # III. SOLVE INVERSE KINEMATICS
    # ----------------------------------------------------------------------------------------------
    dq_cmd, p_ik, quat_cur, _ = ikQP.ik_qp_velocity_step_pose(
        ee_name=FRAME_CONTROL,
        p_des=fsm_ctx.pose_ref,
        quat_des=fsm_ctx.quat_ref,
        q_current=q,
        q_min=Q_MIN, q_max=Q_MAX,
        dq_min=DQ_MIN, dq_max=DQ_MAX,
        dt=DT_CTRL,
        kp_pos=2 / DT_CTRL,
        kp_ori=0.15 * (2 / DT_CTRL),
        w_pos=pos_weight, w_ori=orientation_weight,
        lam=fsm_ctx.joint_vel_weight,
    )

    # ----------------------------------------------------------------------------------------------
    # IV. SEND COMMANDS & SYNC
    # ----------------------------------------------------------------------------------------------
    q = ikQP.send_position_command(dq_cmd=dq_cmd, dt_ctrl=DT_CTRL, q_current=q)

    elapsed = time.time() - loop_start_time
    if elapsed < DT_CTRL:
        time.sleep(DT_CTRL - elapsed)

logger.info("[SHUTDOWN] | Waiting to shutdown ...")
time.sleep(10.0)  # Allow observation before ending
logger.info("[SHUTDOWN] | Control loop ended.")
robot_interface.close()