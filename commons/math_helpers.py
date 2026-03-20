import numpy as np
from typing import Tuple

class MathHelpers:
    # ----------------------------------------------------------------------------------------------
    # HELPER METHODS (MATH & QUATERNIONS)
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
    def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def align_quaternion_sign(reference_quat, candidate_quat):
        """
        Align candidate_quat to the same quaternion hemisphere as reference_quat
        to avoid discontinuities caused by the q and -q representation.
        """
        if np.dot(reference_quat, candidate_quat) < 0:
            return -candidate_quat
        return candidate_quat
