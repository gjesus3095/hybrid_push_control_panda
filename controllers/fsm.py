import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Set, Dict, Any
from hybrid_push_control_panda.commons.logger import logger

# ==================================================================================================
# FSM CLASSES
# ==================================================================================================
@dataclass(frozen=True, slots=True)
class FSMConfig:
    """
    Compact FSM Configuration.
    Units: target_x/y in meters. Geometries are unique string identifiers.
    """
    left_ee: Set[str]
    right_ee: Set[str]
    objects: Set[str]
    ee_covers: Set[str]
    target_x: float
    target_y: float
    delta_tol: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FSMConfig":
        """Maps legacy dictionary keys to dataclass fields."""
        return cls(
            left_ee=set(data.get("LEFT_EE_GEOMS", [])),
            right_ee=set(data.get("RIGHT_EE_GEOMS", [])),
            objects=set(data.get("OBJECT_NAMES", [])),
            ee_covers=set(data.get("EE_COVER_GEOMS", [])),
            target_x=float(data.get("x_displacement", 0.0)),
            target_y=float(data.get("y_displacement", 0.0)),
            delta_tol =float(data.get("delta_tol", 0.01))
        )

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


@dataclass(frozen=True)
class StateTransitionParams:
    """Immutable physical parameters and timeouts for the FSM.
    Parameters:
    - f_preload: Force in Newtons for the preload phase to ensure stable grip before pushing.
    - f_push: Target force in Newtons along Z-axis during push phase (positive value means pushing down).
    - seek_contact_timeout_s: Time in seconds to wait for contact detection before transitioning to recovery.
    - recover_timeout_s: Time in seconds to wait for contact detection during recovery before stopping.
    - preload_timeout_s: Time in seconds to wait for stable grip confirmation before transitioning to recovery.
    - push_timeout_s: Time in seconds to wait for achieving target displacement during push before stopping.
    """
    f_preload: float = 10.0
    f_push: float = 20.0
    seek_contact_timeout_s: float = 2.0
    recover_timeout_s: float = 2.0
    preload_timeout_s: float = 5.0
    push_timeout_s: float = 30.0

class FsmContext:
    """Holds shared live state and interfaces across FSM states."""
    def __init__(self, fsm_config: FSMConfig, state_transition_params: StateTransitionParams = StateTransitionParams()):
        # Configuration (Static)
        self.config = fsm_config
        self.params = state_transition_params

        # Motion References (Dynamic)
        self.t_in_state = 0.0
        self.pose_ref = None
        self.quat_ref = None
        self.wrench_ref = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.joint_vel_weight = 1e5

        # Live State Variables
        self.p_start_regulation = None
        self.curr_p = None
        self.curr_quat = None

        # Hardware/Software Interfaces (Injected later)
        self.interface: Any = None
        self.ik: Any = None
        self.admittance: Any = None
        self.lp_filter: Any = None


class FiniteStateMachine:
    """ Implements the FSM logic for the pushing task."""
    def __init__(self, ctx: FsmContext):
        self.fsm_ctx = ctx
        self.fsm_config = ctx.config
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
            ctx.wrench_ref = (0, 0, ctx.params.f_preload, 0, 0, 0)
            # Original logic from CLOSE_GRIPPER
            ctx.interface.set_control({"actuator8": 0})
        elif state == State.PUSH:
            ctx.joint_vel_weight = 1e8
            ctx.wrench_ref = (0, 0, ctx.params.f_push, 0, 0, 0)
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
                if (pair & self.fsm_config.left_ee) and (pair & self.fsm_config.objects):
                    left_touched_geoms |= (pair & self.fsm_config.left_ee)
                if (pair & self.fsm_config.right_ee) and (pair & self.fsm_config.objects):
                    right_touched_geoms |= (pair & self.fsm_config.right_ee)

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
                if (pair & self.fsm_config.ee_covers) and (pair & self.fsm_config.objects):
                    measured_force_z = contact['force_world'][2]
                    break

            # Filtering the force
            filtered_force = ctx.lp_filter.filter(x=measured_force_z, dt=dt)

            # 2. Admittance Control Step
            fz_err = ctx.params.f_push - filtered_force
            dz_adj, v, z = ctx.admittance.step(f_ext=fz_err, dt=dt)

            # 3. Update Target (Move X/Y, Regulate Z)
            ctx.pose_ref = ctx.p_start_regulation + np.array([self.fsm_config.target_x, self.fsm_config.target_y, -dz_adj])

            # -- Transition Condition [f3]: Displacement Reached --
            dist_moved_x = ctx.curr_p[0] - ctx.p_start_regulation[0]
            dist_moved_y = ctx.curr_p[1] - ctx.p_start_regulation[1]

            x_reached = ( dist_moved_x >= self.fsm_config.target_x - self.fsm_config.delta_tol if self.fsm_config.target_x >= 0 else dist_moved_x <= self.fsm_config.target_x + self.fsm_config.delta_tol)
            y_reached = (dist_moved_y >= self.fsm_config.target_y - self.fsm_config.delta_tol if self.fsm_config.target_y >= 0 else dist_moved_y <= self.fsm_config.target_y + self.fsm_config.delta_tol )

            logger.debug(f"[PUSH] dX: {dist_moved_x:.4f}/{self.fsm_config.target_x} | dY: {dist_moved_y:.4f}/{self.fsm_config.target_y}| Regulating Z: {dz_adj:.4f}m"
                         f"| Controlled Force: {filtered_force:.2f}/{ctx.params.f_push}N")

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