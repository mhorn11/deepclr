import numpy as np
import pytest

from deepclr.icp import ICPAlgorithm, ICPRegistration


SOURCE = np.random.rand(100, 3)
TEMPLATE = SOURCE + 0.02

MAX_DISTANCE = 0.2
NEIGHBOR_RADIUS = 0.01
MAX_NN = 30

ALGORITHMS = [ICPAlgorithm.ICP_PO2PO, ICPAlgorithm.ICP_PO2PL, ICPAlgorithm.GICP]


@pytest.mark.parametrize('algorithm', ALGORITHMS)
def test_registration(algorithm: ICPAlgorithm) -> None:
    registration = ICPRegistration(algorithm, max_distance=MAX_DISTANCE, neighbor_radius=NEIGHBOR_RADIUS, max_nn=MAX_NN)
    t = registration.prepare(TEMPLATE)
    s = registration.prepare(SOURCE)
    m = registration.register(t, s)
    assert isinstance(m, np.ndarray)
    assert m.shape == (4, 4)
    np.testing.assert_equal(m[3, :], [0, 0, 0, 1])
