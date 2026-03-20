"""
Project: Controlled Push IKQP
Author: Gonzalo Meza
Date: January 2026
Description:
    Main execution script for the Franka Emika Panda push task.
    Implements a Finite State Machine (FSM) to handle approach,
    contact detection, and hybrid force/position control.
License: No License
"""
import time
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Custom Imports
from hybrid_push_control_panda.simulation.mujoco_interface import MujocoInterface
from hybrid_push_control_panda.commons.logger import logger
from hybrid_push_control_panda.controllers.ik_qp import InverseKinematicsController, AdmittanceFilter
from hybrid_push_control_panda.controllers.ik_qp import LowPassFilter, AdmittanceFilterParams
from hybrid_push_control_panda.controllers.fsm import FiniteStateMachine, FsmContext, State, StateTransitionParams, FSMConfig
from hybrid_push_control_panda.commons.math_helpers import MathHelpers as math_helpers

# ==================================================================================================
# 1. CONFIGURATION & CONSTANTS
# ==================================================================================================
# Path to this file → project root → target file
BASE_DIR = Path(__file__).resolve().parents[1]   # adjust depth if needed
XML_FILE = str(BASE_DIR / "hybrid_push_control_panda"/ "model" / "scene.xml")

if not Path(XML_FILE).is_file():
    raise FileNotFoundError(f"XML file not found: {XML_FILE}. Check the path and adjust BASE_DIR if necessary.")

# --- Colliding geometries definition ---
LEFT_EE_GEOMS   = ["finger_l1", "finger_l2", "finger_l3", "finger_l4", "finger_l5"]
RIGHT_EE_GEOMS  = ["finger_r1", "finger_r2", "finger_r3", "finger_r4", "finger_r5"]
EE_COVER_GEOMS  = ["hand_cover0", "hand_cover1", "hand_cover2", "hand_cover3", "hand_cover4", "hand_cover5"]
OBJECT_NAMES    = ["box_geom"]

# --- Control Parameters ---
N_DOFS  = 7  # Set all degrees of freedom to control
DT_CTRL = 0.01  # Control timestep (100Hz)
FRAME_CONTROL = "hand" # Control frame (end-effector) for IK and force control

# Force Regulation Targets
DELTA_X_TARGET  = 0.20    # <================= Desired displacement along X during force reg
DELTA_Y_TARGET  = -0.05   # <================= Desired displacement along Y during force reg
DELTA_TOL       = 3e-3    # <================= Tolerance for reaching target (Control error + contact disturbances)
Z_FORCE         = 20.0    # <================= Target force along Z during push phase (as logged originally)

# ---  Define Orientation Targets to Push the Object (world frame ref) ---
rotation_def = "zyx"  # Define the rotation order for Euler angles (yaw-pitch-roll)
ee_ori_target = np.deg2rad([0, 180, 90])  # <================= Desired orientation for pushing (Euler angles in degrees, converted to radians)

# --- Initial Configuration ---
# Joint limits (radians)
Q_MIN = np.array([-np.pi, -1.7628, -np.pi, -3.0718, -np.pi, -0.0175, -np.pi])
Q_MAX = np.array([np.pi, 1.7628, np.pi, -0.0698, np.pi, 3.7525, np.pi])

# Velocity limits
DQ_MIN = -20.0 * np.ones(N_DOFS)
DQ_MAX = 20.0 * np.ones(N_DOFS)

# Initial configuration for simulation (joint angles in radians)
q_init          = np.zeros(N_DOFS)                # <================= Initial arm configuration (radians)
# Initial commands
q_arm_init_cmd  = [0.0, 0, 0, -1.3, 0, 1.51, 0]   # <================= Initial arm command (radians)
q_ee_init_cmd   = [255]                           # <================= Initial gripper command opening (0-255, where 255 is fully open for Franka gripper)
q_cmd  = np.array([q_arm_init_cmd + q_ee_init_cmd]).flatten()  # Combined initial command for arm and gripper


# Initial IK weights
joint_vel_weight = 1e5
pos_weight, orientation_weight = 1, 1

# ==================================================================================================
# 2. INITIALIZATION
# ==================================================================================================
# --- MuJoCo Interface Setup ---
geometry_contact_names = LEFT_EE_GEOMS + RIGHT_EE_GEOMS + EE_COVER_GEOMS
robot_interface = MujocoInterface(XML_FILE, geometry_contact_names)

# --- Retrieve Actuator and Joint Names ---
robot_actuator_names = robot_interface.get_actuators_names()
ee_name = robot_actuator_names[-1]
robot_joint_names = robot_interface.get_joint_names()

logger.info(f"[INIT] | Joints: {len(robot_joint_names)} | Actuators: {len(robot_actuator_names)} | EE: {ee_name}")

# --- Send Initial Hardware Command ---
init_joint_position = {n: v for n, v in zip(robot_joint_names, q_init)}
init_actuator_command = {n: v for n, v in zip(robot_actuator_names, q_cmd)}

robot_interface.send_joints_position_command(init_joint_position) # Set initial joint positions to zero for safety before moving to the desired configuration
robot_interface.set_control(init_actuator_command) # Move to the desired initial configuration (arm + gripper)

# Apply rotation (global frame)
R_euler = R.from_euler(rotation_def, ee_ori_target).as_matrix()
pick_ori_des = R.from_matrix(R_euler).as_quat()
pick_ori_des = math_helpers.change_quaternion_xyzw(pick_ori_des)
pick_ori_des = math_helpers.normalize_quaternion(pick_ori_des)

# --- 3. Set Initial Sim State ---
logger.info("[INIT] | Setting initial configuration and stabilizing...")
logger.info(f"[INIT] | Setting displacement targets for force regulation: X: {DELTA_X_TARGET:.2f}m | Y: {DELTA_Y_TARGET:.2f}m")
logger.info(f"[INIT] | Setting force regulation target Z-axis: {Z_FORCE} N") # Target Z force as logged originally
time.sleep(2.0)  # Allow physics to settle

# ==================================================================================================
# 3. INITIALIZATION OF FSM & CONTEXT
# ==================================================================================================

# 1. Prepare Configuration (Geometry & Targets)
# Using the class factory ensures all types are validated before the FSM starts.
fsm_config = FSMConfig.from_dict({
    "LEFT_EE_GEOMS":   LEFT_EE_GEOMS,
    "RIGHT_EE_GEOMS":  RIGHT_EE_GEOMS,
    "OBJECT_NAMES":    OBJECT_NAMES,
    "EE_COVER_GEOMS":  EE_COVER_GEOMS,
    "x_displacement":  DELTA_X_TARGET,
    "y_displacement":  DELTA_Y_TARGET,
    "delta_tol":       DELTA_TOL,
})

# 2. Define Physical Behavior (Forces & Timeouts)
transition_params = StateTransitionParams(f_push=Z_FORCE)

# 3. Assemble the Context
# This object acts as the 'Single Source of Truth' for the entire task.
fsm_ctx = FsmContext(
    fsm_config=fsm_config,
    state_transition_params=transition_params
)

# 4. Inject Hardware/Software Interfaces
# --- Controller Setup ---
ikQP = InverseKinematicsController(robot_interface, n_dofs=N_DOFS)
params = AdmittanceFilterParams(m=2.0, d=200.0, k=1, max_x=0.3, max_v=0.2, max_a=2.0)

fsm_ctx.interface  = robot_interface
fsm_ctx.ik         = ikQP
fsm_ctx.lp_filter  = LowPassFilter(cutoff_hz=50)
fsm_ctx.quat_ref   = pick_ori_des
fsm_ctx.admittance = AdmittanceFilter(params, x0=0.0, v0=0.0)

# 5. Initialize & Boot the FSM
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
    fsm_ctx.curr_quat = ikQP.normalize_quaternion(quat_cur)

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