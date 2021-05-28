import abc
from typing import Dict, Optional, Tuple
import warnings

import torch
import torch.nn as nn


class BaseModel(nn.Module, metaclass=abc.ABCMeta):
    """BaseModel for all point cloud registration models."""

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_input_dim(self) -> int:
        """Return expected dimension for input points."""
        raise NotImplementedError

    @abc.abstractmethod
    def has_loss(self) -> bool:
        """Return of forward method provides the loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def get_loss_weights(self) -> Dict:
        """Return dictionary containing the current loss weights."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, is_feat: bool = False, m: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None, debug: bool = False)\
            -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Predict registration for a point cloud batch.
            :param x: Point cloud batch
            :param is_feat: Input x is already preprocessed with features
            :param m: Augmentation matrices
            :param y: Optional ground truth
            :param debug: Return debug information
            :return: Predicted transformations, Loss, Debug information
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cloud_features(self, x: torch.Tensor, m: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Preprocess features for point cloud batch.
            :param x: Point cloud batch
            :param m: Augmentation matrix
            :return: Point cloud batch with features
        """
        raise NotImplementedError


class ModelInferenceHelper:
    """Helper class for sequential and non-sequential model inference."""
    def __init__(self, model: BaseModel, is_sequential: bool = False):
        self._model = model
        self._input_dim = model.get_input_dim()
        self._model.eval()
        self._is_sequential = is_sequential
        self._state: Optional[torch.Tensor] = None

    def has_state(self) -> bool:
        return self._state is not None

    def reset_state(self) -> None:
        """Reset internal state, e.g., in case the next sequence is started."""
        self._state = None

    def predict(self, source: torch.Tensor, template: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Predict transform.
            :param source: Single source point cloud.
            :param template: Single template point cloud for non-sequential prediction.
            :return: Batch with both clouds.
        """
        # truncate input
        if source.shape[1] > self._input_dim:
            warnings.warn(f"Truncate source point cloud from dimension {source.shape[1]} "
                          f"to required dimension {self._input_dim}.")
            source = source[:, :self._input_dim]
        elif source.shape[1] < self._input_dim:
            raise RuntimeError("Wrong point dimension in source.")

        if template is not None:
            if template.shape[1] > self._input_dim:
                warnings.warn(f"Truncate template point cloud from dimension {template.shape[1]} "
                              f"to required dimension {self._input_dim}.")
                template = template[:, :self._input_dim]
            elif template.shape[1] < self._input_dim:
                raise RuntimeError("Wrong point dimension in template.")

        # inference
        with torch.no_grad():
            if self._is_sequential:
                if template is not None:
                    raise RuntimeError("Only the source cloud is required for sequential prediction.")

                source = self._model.cloud_features(source.unsqueeze(0))[0, ...]

                if self._state is None:
                    # first call
                    self._state = source
                    return None
                else:
                    # subsequent call
                    x = self.stack(self._state, source)
                    y, _, _ = self._model.forward(x, is_feat=True)
                    self._state = source
                    return y[0, :]

            else:
                if template is None:
                    raise RuntimeError("Source and template clouds are required for non-sequential prediction.")

                x = self.stack(template, source)
                y, _, _ = self._model.forward(x, is_feat=False)
                return y[0, :]

    @staticmethod
    def stack(template: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """
        Stack two point clouds to a batch.
            :param template: Template point cloud (with or without features)
            :param source: Source point cloud (with or without features)
            :return: Batch with both clouds.
        """
        if template.shape[0] < source.shape[0]:
            perm = torch.randperm(source.shape[0])[:template.shape[0]]
            source = source[perm, :]
        elif template.shape[0] > source.shape[0]:
            perm = torch.randperm(template.shape[0])[:source.shape[0]]
            template = template[perm, :]
        return torch.stack((template, source), 0)
