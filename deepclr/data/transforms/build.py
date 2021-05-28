import torchvision.transforms

from ...config.config import Config
from .transforms import FarthestPointSampling, PointNoise, RandomErasing, RandomTransform, RangeSelection, \
    RemoveTransform, SystematicErasing, TruncateDimension
from .utils import NoiseType


def build_transform(cfg: Config, is_training: bool = True) -> torchvision.transforms.Compose:
    """Create transform composition from config."""
    input_dim = cfg.model.input_dim
    point_dim = cfg.model.point_dim

    cfg = cfg.transforms
    if is_training or cfg.on_validation:
        if cfg.nth_point_random:
            nth_point_start = -1
        else:
            nth_point_start = 0

        transform = torchvision.transforms.Compose([
            TruncateDimension(input_dim),
            SystematicErasing(cfg.nth_point, start=nth_point_start),
            RangeSelection(cfg.min_range, cfg.max_range, dim=point_dim),
            RandomErasing(cfg.keep_probability, cfg.max_points),
            FarthestPointSampling(cfg.fps, dim=point_dim),
            RemoveTransform(cfg.remove_transform, dim=point_dim),
            RandomTransform(cfg.translation_noise.scale, cfg.rotation_noise_deg.scale, dim=point_dim,
                            translation_noise_type=cfg.translation_noise.type,
                            rotation_noise_deg_type=cfg.rotation_noise_deg.type),
            PointNoise(cfg.point_noise.scale, noise_type=NoiseType[cfg.point_noise.type.upper()],
                       target_only=cfg.point_noise.target_only, dim=point_dim)
        ])
    else:
        transform = torchvision.transforms.Compose([
            TruncateDimension(input_dim),
            SystematicErasing(cfg.nth_point, start=0),
            RangeSelection(cfg.min_range, cfg.max_range, dim=point_dim),
            RandomErasing(cfg.keep_probability, cfg.max_points),
            FarthestPointSampling(cfg.fps, dim=point_dim),
        ])
    return transform
