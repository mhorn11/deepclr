from enum import auto
from typing import List, Optional, Tuple

import numpy as np
import transforms3d

from ..config.config import ConfigEnum


class LabelType(ConfigEnum):
    """Available label types and the respective label transformations."""
    POSE3D_EULER = auto()
    POSE3D_QUAT = auto()
    POSE3D_DUAL_QUAT = auto()

    @property
    def dim(self) -> int:
        if self == self.POSE3D_EULER:
            return 6
        if self == self.POSE3D_QUAT:
            return 7
        if self == self.POSE3D_DUAL_QUAT:
            return 8
        raise NotImplementedError("LabelType '{}' not implemented".format(self))

    @property
    def names(self):
        if self == self.POSE3D_EULER:
            return ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        if self == self.POSE3D_QUAT:
            return ['pos_x', 'pos_y', 'pos_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        if self == self.POSE3D_DUAL_QUAT:
            return ['real_w', 'real_x', 'real_y', 'real_z', 'dual_w', 'dual_x', 'dual_y', 'dual_z']
        raise NotImplementedError("LabelType '{}' not implemented".format(self))

    @property
    def bias(self) -> Optional[List[float]]:
        if self == self.POSE3D_EULER:
            return None
        if self == self.POSE3D_QUAT:
            return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        if self == self.POSE3D_DUAL_QUAT:
            return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        raise NotImplementedError("LabelType '{}' not implemented".format(self))

    @staticmethod
    def _dqnormalize(real: np.ndarray, dual: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        real_norm = np.sqrt(np.dot(real, real)) + eps
        real = real / real_norm
        dual = dual / real_norm
        return real, dual

    def from_matrix(self, data: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        if self == self.POSE3D_EULER:
            trans, rot, _, _ = transforms3d.affines.decompose(data)
            roll, pitch, yaw = transforms3d.euler.mat2euler(rot, axes='sxyz')
            label = np.array([trans[0], trans[1], trans[2], np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)])

        elif self == self.POSE3D_QUAT:
            trans, rot, _, _ = transforms3d.affines.decompose(data)
            quat = transforms3d.quaternions.mat2quat(rot)
            label = np.array([trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]])

        elif self == self.POSE3D_DUAL_QUAT:
            t, r, _, _ = transforms3d.affines.decompose(data)
            real = transforms3d.quaternions.mat2quat(r)
            dual = 0.5 * transforms3d.quaternions.qmult(np.array([0, t[0], t[1], t[2]]), real)
            label = np.array([real[0], real[1], real[2], real[3], dual[0], dual[1], dual[2], dual[3]])

        else:
            raise NotImplementedError("LabelType '{}' not implemented".format(self))

        if scale is not None:
            label *= scale

        return label

    def to_matrix(self, label: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        if scale is not None:
            label /= scale

        if self == self.POSE3D_EULER:
            trans = label[:3]
            rot = transforms3d.euler.euler2mat(np.deg2rad(label[3]), np.deg2rad(label[4]), np.deg2rad(label[5]),
                                               axes='sxyz')
            return transforms3d.affines.compose(trans, rot, np.ones(3))

        elif self == self.POSE3D_QUAT:
            trans = label[:3]
            rot = transforms3d.quaternions.quat2mat(label[3:])
            return transforms3d.affines.compose(trans, rot, np.ones(3))

        elif self == self.POSE3D_DUAL_QUAT:
            real, dual = self._dqnormalize(label[:4], label[4:])
            m = np.eye(4)
            m[:3, :3] = transforms3d.quaternions.quat2mat(real)
            t = 2.0 * transforms3d.quaternions.qmult(dual, transforms3d.quaternions.qconjugate(real))
            m[:3, 3] = t[1:]
            return m
        else:
            raise NotImplementedError("LabelType '{}' not implemented".format(self))
