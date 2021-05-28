from enum import auto
from typing import Any

import gicp
import numpy as np
import open3d as o3d

from ..config.config import ConfigEnum


class ICPAlgorithm(ConfigEnum):
    ICP_PO2PO = auto()  # ICP with point-to-point metric
    ICP_PO2PL = auto()  # ICP with point-to-plane metric
    GICP = auto()  # Generalized ICP


class ICPRegistration:
    """Prepare and register point clouds with different ICP variants."""
    def __init__(self, algorithm: ICPAlgorithm, max_distance: float, neighbor_radius: float, max_nn: int):
        self._algorithm = algorithm
        self._max_distance = max_distance
        self._neighbor_radius = neighbor_radius
        self._max_nn = max_nn

    def prepare(self, cloud: np.ndarray) -> Any:
        """Convert point clouds to required format."""
        if self._algorithm == ICPAlgorithm.ICP_PO2PO or self._algorithm == ICPAlgorithm.ICP_PO2PL:
            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(cloud)
            return o3d_cloud

        elif self._algorithm == ICPAlgorithm.GICP:
            gicp_cloud = gicp.prepare(cloud)
            return gicp_cloud

        else:
            raise TypeError("ICPAlgorithm not supported")

    def register(self, template: Any, source: Any) -> np.ndarray:
        """Register converted point clouds."""
        if self._algorithm == ICPAlgorithm.ICP_PO2PO:
            reg = o3d.pipelines.registration.registration_icp(
                source, template, self._max_distance, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            return np.array(reg.transformation)

        elif self._algorithm == ICPAlgorithm.ICP_PO2PL:
            # estimate plane normals
            template.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._neighbor_radius, max_nn=self._max_nn))
            source.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self._neighbor_radius, max_nn=self._max_nn))

            # estimate transform
            reg = o3d.pipelines.registration.registration_icp(
                source, template, self._max_distance, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            return np.array(reg.transformation)

        elif self._algorithm == ICPAlgorithm.GICP:
            m = gicp.gicp(source, template)
            return np.array(m)

        else:
            raise TypeError("ICPAlgorithm not supported")
