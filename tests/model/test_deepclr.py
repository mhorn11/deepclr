import os.path as osp

import torch

from deepclr.config import load_model_config
from deepclr.models import build_model
from deepclr.models.deepclr import SetAbstraction, MotionEmbedding, OutputSimple


CLOUD_COUNT = 5
POINT_COUNT = 96

CONFIG_FILE = osp.join(osp.dirname(osp.abspath(__file__)), 'deepclr.yaml')
CONFIG = load_model_config(CONFIG_FILE, '')


def test_layers():
    # data
    clouds_internal = torch.rand(CLOUD_COUNT * 2, CONFIG.input_dim, POINT_COUNT).cuda()

    # SetAbstraction
    set_abstraction_layer = SetAbstraction(input_dim=CONFIG.input_dim, point_dim=CONFIG.point_dim,
                                           **CONFIG.params.cloud_features.params).cuda()
    set_abstraction = set_abstraction_layer(clouds_internal)
    assert set_abstraction.shape == (CLOUD_COUNT * 2, 67, 1024)

    # MotionEmbedding
    motion_embedding_layer = MotionEmbedding(input_dim=set_abstraction_layer.output_dim(), point_dim=CONFIG.point_dim,
                                             **CONFIG.params.merge.params).cuda()
    motion_embedding = motion_embedding_layer(set_abstraction)
    assert motion_embedding.shape == (CLOUD_COUNT, 259, 1024)

    # OutputBase
    output_simple_layer = OutputSimple(input_dim=motion_embedding_layer.output_dim(), label_type=CONFIG.label_type,
                                       **CONFIG.params.output.params).cuda()
    output_simple = output_simple_layer(motion_embedding)
    assert output_simple.shape == (CLOUD_COUNT, CONFIG.label_type.dim)


def test_model():
    # data
    clouds = torch.rand(CLOUD_COUNT * 2, POINT_COUNT, CONFIG.input_dim).cuda()
    y = torch.rand(CLOUD_COUNT, CONFIG.label_type.dim).cuda()

    # model
    model = build_model(CONFIG).cuda()

    with torch.no_grad():
        # inference
        y_pred1, loss1, _ = model(clouds, y=y)
        assert y_pred1.shape == (CLOUD_COUNT, CONFIG.label_type.dim)
        assert loss1.shape == ()

        clouds_feat = model.cloud_features(clouds)
        y_pred2, loss2, _ = model(clouds_feat, y=y, is_feat=True)
        assert y_pred2.shape == (CLOUD_COUNT, CONFIG.label_type.dim)
        assert loss2.shape == ()
