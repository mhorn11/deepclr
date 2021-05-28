from enum import auto
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from ..config.config import Config, ConfigEnum
from ..data.labels import LabelType
from ..utils.tensor import prepare_tensor
from .quaternion import qconjugate, qmult


MetricFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GenericMetricFunction = Callable[[torch.Tensor, torch.Tensor, Optional[str]], torch.Tensor]


def _apply_reduction(x: torch.Tensor, reduction: Optional[str]) -> torch.Tensor:
    if reduction is None or reduction == 'none':
        return x
    if reduction == 'mean':
        return torch.mean(x)
    elif reduction == 'sum':
        return torch.sum(x)
    else:
        raise RuntimeError(f"Unsupported reduction '{reduction}'")


def _quat_norm(source: torch.Tensor, _target: torch.Tensor, label_type: LabelType,
               reduction: Optional[str] = 'mean') -> torch.Tensor:
    """Quaternion norm of source tensor."""
    if label_type == LabelType.POSE3D_QUAT:
        source_norm = torch.norm(source[:, 3:], p=2, dim=1, keepdim=True)
    elif label_type == LabelType.POSE3D_DUAL_QUAT:
        source_norm = torch.norm(source[:, :4], p=2, dim=1, keepdim=True)
    else:
        raise RuntimeError("Unsupported label type for this loss type")

    return _apply_reduction(source_norm, reduction)


def _normalize(x: torch.Tensor, label_type: LabelType, eps: float = 1e-8) -> torch.Tensor:
    if label_type == LabelType.POSE3D_QUAT:
        x_norm = torch.norm(x[:, 3:], p=2, dim=1, keepdim=True) + eps
        x = torch.cat((x[:, :3], x[:, 3:] / x_norm), dim=1)
        return x
    elif label_type == LabelType.POSE3D_DUAL_QUAT:
        x_norm = torch.norm(x[:, :4], p=2, dim=1, keepdim=True) + eps
        x = x / x_norm
        return x
    else:
        raise RuntimeError("Unsupported label type for normalization")


def trans_loss(source: torch.Tensor, target: torch.Tensor, label_type: LabelType, p: int = 2,
               reduction: Optional[str] = 'mean', eps: float = 1e-8) -> torch.Tensor:
    """Translation (translation directly from label or dual quaternion vector) loss."""
    if label_type == LabelType.POSE3D_EULER or label_type == LabelType.POSE3D_QUAT:
        source_trans = source[:, :3]
        target_trans = target[:, :3]

    elif label_type == LabelType.POSE3D_DUAL_QUAT:
        source = _normalize(source, label_type, eps)
        target = _normalize(target, label_type, eps)
        source_trans = source[:, 4:]
        target_trans = target[:, 4:]

    else:
        raise RuntimeError("Unsupported label type for this loss type.")

    loss = torch.norm(source_trans - target_trans, dim=1, p=p, keepdim=True)
    return _apply_reduction(loss, reduction)


def trans_3d_loss(source: torch.Tensor, target: torch.Tensor, label_type: LabelType, p: int = 2,
                  reduction: Optional[str] = 'mean', eps: float = 1e-8) -> torch.Tensor:
    """Translation in 3D coordinates [x, y, z] loss."""
    if label_type == LabelType.POSE3D_EULER or label_type == LabelType.POSE3D_QUAT:
        source_trans = source[:, :3]
        target_trans = target[:, :3]

    elif label_type == LabelType.POSE3D_DUAL_QUAT:
        # normalize dual quaternion
        source = _normalize(source, label_type, eps)
        target = _normalize(target, label_type, eps)

        # convert dual quaternion to translation vector
        source_trans_quat = 2.0 * qmult(source[:, 4:], qconjugate(source[:, :4]))
        target_trans_quat = 2.0 * qmult(target[:, 4:], qconjugate(target[:, :4]))
        source_trans = source_trans_quat[:, 1:]
        target_trans = target_trans_quat[:, 1:]

    else:
        raise RuntimeError("Unsupported label type for this loss type.")

    loss = torch.norm(source_trans - target_trans, dim=1, p=p, keepdim=True)
    return _apply_reduction(loss, reduction)


def dual_loss(source: torch.Tensor, target: torch.Tensor, label_type: LabelType, p: int = 2,
              reduction: Optional[str] = 'mean', eps: float = 1e-8) -> torch.Tensor:
    """Dual quaternion vector loss."""
    if label_type == LabelType.POSE3D_QUAT:
        # translation quaternion
        source_trans_quat = source.new_zeros(source.shape[0], 4)
        source_trans_quat[:, 1:] = source[:, :3]
        target_trans_quat = target.new_zeros(target.shape[0], 4)
        target_trans_quat[:, 1:] = target[:, :3]

        # dual quaternions
        source_dual = 0.5 * qmult(source_trans_quat, source[:, 3:])
        target_dual = 0.5 * qmult(target_trans_quat, target[:, 3:])

    elif label_type == LabelType.POSE3D_DUAL_QUAT:
        source = _normalize(source, label_type, eps)
        target = _normalize(target, label_type, eps)
        source_dual = source[:, 4:]
        target_dual = target[:, 4:]

    else:
        raise RuntimeError("Unsupported label type for this loss type")

    loss = torch.norm(source_dual - target_dual, dim=1, p=p, keepdim=True)
    return _apply_reduction(loss, reduction)


def rot_loss(source: torch.Tensor, target: torch.Tensor, label_type: LabelType, p: int = 2,
             reduction: Optional[str] = 'mean', eps: float = 1e-8) -> torch.Tensor:
    """Rotation vector (either euler angles or quaternion vector) loss."""
    if label_type == LabelType.POSE3D_EULER:
        source_rot = source[:, 3:]
        target_rot = target[:, 3:]

    elif label_type == LabelType.POSE3D_QUAT:
        source = _normalize(source, label_type, eps)
        target = _normalize(target, label_type, eps)
        source_rot = source[:, 3:]
        target_rot = target[:, 3:]

    elif label_type == LabelType.POSE3D_DUAL_QUAT:
        source = _normalize(source, label_type, eps)
        target = _normalize(target, label_type, eps)
        source_rot = source[:, :4]
        target_rot = target[:, :4]

    else:
        raise RuntimeError("Unsupported label type for this loss type")

    loss = torch.norm(source_rot - target_rot, dim=1, p=p, keepdim=True)
    return _apply_reduction(loss, reduction)


def quat_norm_loss(source: torch.Tensor, target: torch.Tensor, label_type: LabelType,
                   reduction: Optional[str] = 'mean') -> torch.Tensor:
    """Quaternion norm loss."""
    if label_type != LabelType.POSE3D_QUAT and label_type != LabelType.POSE3D_DUAL_QUAT:
        raise RuntimeError("Unsupported label type for this loss type.")

    source_norm = _quat_norm(source, target, label_type, reduction=None)
    loss = torch.pow(1.0 - source_norm, 2)

    return _apply_reduction(loss, reduction)


def dual_constraint_loss(source: torch.Tensor, _target: torch.Tensor, label_type: LabelType,
                         reduction: Optional[str] = 'mean', eps: float = 1e-8) -> torch.Tensor:
    """Dual quaternion constraint loss."""
    if label_type != LabelType.POSE3D_DUAL_QUAT:
        raise RuntimeError("Unsupported label type for this loss type.")

    source = _normalize(source, label_type, eps)
    source_trans_quat = 2.0 * qmult(source[:, 4:], qconjugate(source[:, :4]))
    loss = torch.pow(source_trans_quat[:, 0], 2)

    return _apply_reduction(loss, reduction)


def _weighted_loss(metric_fn: GenericMetricFunction, source: torch.Tensor, target: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """Weighted sum of loss function output."""
    ret = metric_fn(source, target, 'none')
    return torch.sum(weights * torch.mean(ret, 0))


def _weighted_loss_fn(metric_fn: GenericMetricFunction, weights: Optional[torch.Tensor] = None) -> MetricFunction:
    """Create weighted loss function."""
    if weights is None:
        def func(source, target):
            return metric_fn(source, target, 'mean')
        return func
    else:
        def func(source, target):
            return _weighted_loss(metric_fn, source, target, weights)  # type: ignore
        return func


class MetricType(ConfigEnum):
    """Enum with all available loss types."""
    MAE = auto()
    MSE = auto()
    TRANS = auto()
    TRANS_3D = auto()
    DUAL = auto()
    ROT = auto()
    QUAT_NORM = auto()
    DUAL_CONSTRAINT = auto()

    def fn(self, label_type: LabelType, weights: Optional[torch.Tensor] = None, **kwargs: Any) -> MetricFunction:
        func: Optional[GenericMetricFunction] = None

        if self == MetricType.MAE:
            def func(source, target, red): return F.l1_loss(source, target, reduction=red, **kwargs)
        elif self == MetricType.MSE:
            def func(source, target, red): return F.mse_loss(source, target, reduction=red, **kwargs)
        elif self == MetricType.TRANS:
            def func(source, target, red): return trans_loss(source, target, label_type, reduction=red, **kwargs)
        elif self == MetricType.TRANS_3D:
            def func(source, target, red): return trans_3d_loss(source, target, label_type, reduction=red, **kwargs)
        elif self == MetricType.DUAL:
            def func(source, target, red): return dual_loss(source, target, label_type, reduction=red, **kwargs)
        elif self == MetricType.ROT:
            def func(source, target, red): return rot_loss(source, target, label_type, reduction=red, **kwargs)
        elif self == MetricType.QUAT_NORM:
            def func(source, target, red): return quat_norm_loss(source, target, label_type, reduction=red)
        elif self == MetricType.DUAL_CONSTRAINT:
            def func(source, target, red): return dual_constraint_loss(source, target, label_type, reduction=red)

        if func is not None:
            return _weighted_loss_fn(func, weights)
        else:
            raise NotImplementedError("MetricType '{}' not implemented".format(self))


def get_loss_fn(cfg: Config) -> MetricFunction:
    """Create loss function from config."""
    label_type = cfg.model.label_type

    # weights loss functions
    loss_functions = list()
    for metric_data in cfg.metrics.loss:
        # weights
        weights = metric_data['weights']
        if weights is not None:
            weights = prepare_tensor(torch.FloatTensor(weights), device=cfg.device, non_blocking=False)

        # function
        loss_functions.append(metric_data['type'].fn(label_type, weights=weights, **metric_data['params']))

    # sum weighted loss
    def func(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.stack([f(source, target) for f in loss_functions])
        return torch.sum(loss)

    return func


def get_metric_fns(cfg: Config) -> Dict[str, MetricFunction]:
    """Create metric functions from config."""
    metric_fns = dict()
    for metric_data in [*cfg.metrics.loss, *cfg.metrics.other]:
        params = metric_data['params'] if 'params' in metric_data else dict()
        metric_fns[metric_data['type'].name.lower()] = metric_data['type'].fn(cfg.model.label_type, **params)
    return metric_fns
