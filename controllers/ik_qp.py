import numpy as np
import scipy.sparse as sp
import osqp
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

# ==================================================================================================
# FILTERS FOR MEASUREMENTS
# ==================================================================================================
class LowPassFilter:
    """First-order IIR low-pass filter for scalars."""
    def __init__(self, cutoff_hz: float):
        self.cutoff_hz = cutoff_hz
        self._y: Optional[float] = None

    def reset_filter(self, y0: Optional[float] = None) -> None:
        self._y = y0

    def filter(self, x: float, dt: float) -> float:
        fc = self.cutoff_hz
        if fc is None or fc <= 0:
            return x
        if dt <= 0:
            raise ValueError("dt must be > 0")

        tau = 1.0 / (2.0 * np.pi * fc)
        alpha = dt / (dt + tau)

        if self._y is None:
            self._y = x
        else:
            self._y = (1.0 - alpha) * self._y + alpha * x
        return self._y

# ==================================================================================================
# ADMITTANCE CONTROLLER
# ==================================================================================================
@dataclass
class AdmittanceFilterParams:
    m: float = 1.0                 # [kg] virtual mass
    d: float = 50.0                # [N*s/m] damping
    k: float = 1000.0              # [N/m] stiffness
    max_x: Optional[float] = None  # [m] displacement limit
    max_v: Optional[float] = None  # [m/s] velocity limit
    max_a: Optional[float] = None  # [m/s^2] acceleration limit


class AdmittanceFilter:
    """
    Discrete-time 1D admittance filter:
        m * xdd + d * xd + k * x = F_ext
    State: (x, v). Input: F_ext, dt. Output: (x, v, a).
    Uses semi-implicit (symplectic) Euler: v <- v + a*dt, x <- x + v*dt.
    """

    def __init__(self, params: AdmittanceFilterParams, x0: float = 0.0, v0: float = 0.0):
        if params.m <= 0:
            raise ValueError("m must be > 0")
        if params.d < 0 or params.k < 0:
            raise ValueError("d and k must be >= 0")

        self.p = params
        self.x = x0
        self.v = v0

    def reset(self, x0: float = 0.0, v0: float = 0.0) -> None:
        self.x = x0
        self.v = v0

    def step(self, f_ext: float, dt: float) -> Tuple[float, float, float]:
        """
        Args:
            f_ext: external force input [N]
            dt: timestep [s]
        Returns:
            x: displacement [m]
            v: velocity [m/s]
            a: acceleration [m/s^2]
        """
        if dt <= 0:
            raise ValueError("dt must be > 0")

        # Dynamics
        a = (f_ext - self.p.d * self.v - self.p.k * self.x) / self.p.m

        # Integration
        self.v = self.v + a * dt
        self.x = self.x + self.v * dt

        # If position saturates, kill velocity to avoid “integrating into the wall”
        if self.p.max_x is not None and abs(self.x) >= self.p.max_x:
            self.v = 0.0

        return self.x, self.v, a


# ==================================================================================================
# DIFF INVERSE KINEMATICS CONTROLLER (QP)
# ==================================================================================================

class InverseKinematicsController:
    """
    Differential Inverse Kinematics controller using Quadratic Programming (QP).
    Handles joint limits, velocity limits, and singularity damping.
    """

    def __init__(self, robot_interface, n_dofs: int = 7):
        """
        Args:
            robot_interface: Instance of MujocoInterface handling sim communication.
            n_dofs (int): Number of degrees of freedom to control.
        """
        self.robot = robot_interface
        self.n_dofs = n_dofs

        # Cache joint/actuator names for the controlled subset
        self.joint_names = self.robot.get_joint_names()[:self.n_dofs]
        self.actuator_names = self.robot.get_actuators_names()[:self.n_dofs]

    # ----------------------------------------------------------------------------------------------
    # 1. SOLVERS
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def solve_box_qp(H: np.ndarray, f: np.ndarray,
                     lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
        """
        Solves the box-constrained Quadratic Program:
            min  1/2 x^T H x + f^T x
            s.t. lb <= x <= ub

        Args:
            H (np.ndarray): Hessian matrix (nxn).
            f (np.ndarray): Gradient vector (n,).
            lb (np.ndarray): Lower bound vector (n,).
            ub (np.ndarray): Upper bound vector (n,).

        Returns:
            np.ndarray: Optimal solution vector x.
        """
        n = H.shape[0]

        # Convert to sparse CSC format for OSQP
        P = sp.csc_matrix((H + H.T) * 0.5)
        q = f
        A = sp.eye(n, format="csc")

        # Initialize solver
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, u=ub, l=lb, verbose=False, polish=True)

        res = prob.solve()

        # Check solver status (1=solved, 2=solved_inaccurate)
        if res.info.status_val not in (1, 2):
            raise RuntimeError(f"OSQP Solver failed: {res.info.status}")

        return res.x

    # ----------------------------------------------------------------------------------------------
    # 2. HELPER METHODS (MATH & QUATERNIONS)
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def extract_position_quat_from_pose(pose) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parses pose data into position and quaternion arrays.
        Expected format: (pos(3,), quat(4,))
        """
        if isinstance(pose, (tuple, list)) and len(pose) == 2:
            p = np.asarray(pose[0], dtype=float).reshape(3)
            q = np.asarray(pose[1], dtype=float).reshape(4)
            return p, q
        raise ValueError(f"Unsupported pose format, expected (pos, quat), got {type(pose)}")

    @staticmethod
    def quat_normalize(q: np.ndarray) -> np.ndarray:
        """ Safely normalizes a quaternion. Returns identity if norm is near zero. """
        q = np.asarray(q, dtype=float).reshape(4)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return q / n

    @staticmethod
    def check_quaternion(quat: np.ndarray) -> np.ndarray:
        """
        Enforces a positive scalar component (w >= 0) to ensure uniqueness
        in 'double cover' representation.
        Assumes input format [x, y, z, w].
        """
        if quat[3] < 0:
            return -quat
        return quat

    @staticmethod
    def change_quaternion_xyzw(quat: np.ndarray) -> np.ndarray:
        """
        Permutes quaternion elements.
        NOTE: Based on array logic: [0,1,2,3] -> [1,2,3,0].
        If input is [w, x, y, z] (MuJoCo), output is [x, y, z, w] (SciPy).
        """
        return np.array([quat[1], quat[2], quat[3], quat[0]])

    def quat_error_eps_form(self, q_des: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
        """
        Computes orientation error using the epsilon/eta formulation.

        Args:
            q_des (np.ndarray): Desired quaternion [x, y, z, w].
            q_cur (np.ndarray): Current quaternion [x, y, z, w].

        Returns:
            e_phi (np.ndarray): 3D orientation error vector.
        """
        q_des = self.check_quaternion(q_des)
        q_cur = self.check_quaternion(q_cur)

        # Split Vector (eps) and Scalar (eta) parts
        eps_des = q_des[:3]
        eta_des = q_des[3]

        eps_cur = q_cur[:3]
        eta_cur = q_cur[3]

        # Ensure shortest path (same hemisphere)
        if np.dot(q_des, q_cur) < 0.0:
            eps_des = -eps_des
            eta_des = -eta_des

        # Calculate error: e = eta_c*eps_d - eta_d*eps_c - S(eps_d)eps_c
        e_phi = (
                eta_cur * eps_des
                - eta_des * eps_cur
                - np.cross(eps_des, eps_cur)
        )

        return e_phi

    # ----------------------------------------------------------------------------------------------
    # 3. CORE CONTROL LOGIC
    # ----------------------------------------------------------------------------------------------
    def ik_qp_velocity_step_pose(self, ee_name: str,
                                 p_des: np.ndarray, quat_des: np.ndarray,
                                 q_current: np.ndarray,
                                 q_min: np.ndarray, q_max: np.ndarray,
                                 dq_min: np.ndarray, dq_max: np.ndarray,
                                 dt: float,
                                 kp_pos: float = 150.0, kp_ori: float = 60.0,
                                 w_pos: float = 1.0, w_ori: float = 1.0,
                                 lam: float = 1e-3):
        """
        Computes joint velocities (dq) to reach a desired pose using QP.

        Optimization Objective:
            minimize || W ( (p_dot_desired -J dq) + Kp (p_desired - p_current)) ||^2 + lam ||dq||^2

        Since there is no trajectory generation p_dot_desired = 0, the objective simplifies to:
            minimize || W (-J dq + error_des) ||^2 + lam ||dq
        Subject to:
            dq_min <= dq <= dq_max  (Velocity limits)
            q_min  <= q + dq*dt <= q_max (Position feasibility)

        Args:
            ee_name: Name of the end-effector site/body.
            p_des: Desired position (3,).
            quat_des: Desired orientation quaternion [x,y,z,w].
            q_current: Current joint configuration.
            lam: Damping factor for joint velocities.
        """
        # Initial weights and dimensions
        task_space = 6
        weigh_task_satisfaction = 1e8

        # 1. Get Current State
        pose = self.robot.get_pose(ee_name)
        p, quat_cur = self.extract_position_quat_from_pose(pose)
        quat_cur = self.change_quaternion_xyzw(quat_cur)

        # 2. Jacobian Computation
        J = np.asarray(self.robot.get_jacobian(ee_name), dtype=float)
        # Extract linear (top 3) and angular (bottom 3) rows, limited to n_dofs cols
        J6 = J[:6, :self.n_dofs]

        # 3. Task Space Error & Desired Velocity (PD Control)
        # v_des = Kp * (p_des - p)
        v_des = kp_pos * (np.asarray(p_des) - p)

        # w_des = Ko * orientation_error
        e_ori = self.quat_error_eps_form(quat_des, quat_cur)
        w_des = kp_ori * e_ori

        error_des = np.hstack([v_des, w_des])

        # 4. QP Formulation setup
        # Weighting matrices
        K = np.diag([w_pos] * 3 + [w_ori] * 3)
        W = np.diag([weigh_task_satisfaction] * task_space)  # High weight on task satisfaction

        # Prepare matrices for: 0.5 x^T H x + f^T x
        JW = W @ J6
        bW = K @ error_des

        H = (J6.T @ JW) + lam * np.eye(self.n_dofs)
        f = -(bW.T @ JW).T

        # 5. Constraints (Velocity + Position Safety)
        # Calculate max velocity allowed to stay within position limits in one timestep
        dq_pos_lb = (q_min - q_current) / dt
        dq_pos_ub = (q_max - q_current) / dt

        # Intersect with absolute velocity limits
        lb = np.maximum(dq_min, dq_pos_lb)
        ub = np.minimum(dq_max, dq_pos_ub)
        lb = np.minimum(lb, ub)  # Ensure lb <= ub (safety clamp)

        # 6. Solve
        dq_cmd = self.solve_box_qp(H, f, lb, ub)

        return dq_cmd, p, quat_cur, self.joint_names

    def send_position_command(self, dq_cmd: np.ndarray, dt_ctrl: float, q_current: np.ndarray) -> np.ndarray:
        """
        Integrates velocity command to position and sends to robot interface.
        Returns the commanded joint positions.
        """
        # Euler Integration: q_next = q_curr + dq * dt
        q_cmd = q_current + dt_ctrl * dq_cmd

        # Map to actuator names
        actuator_command = {n: float(v) for n, v in zip(self.actuator_names, q_cmd)}
        self.robot.set_control(actuator_command)

        return q_cmd