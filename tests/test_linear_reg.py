"""Unit Tests for Linear Regression."""
import numpy as np
import numpy.typing as npt
import pytest

from source.linear_regression import LinearRegressionGD


@pytest.fixture
def linear_data()-> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Generate Test Data."""
    x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    y = 2.0 * x + 1.0
    return x, y

def test_init_default() -> None:
    """Tests if values are set correctly on initialization with defaults."""
    model = LinearRegressionGD()
    assert model.lr == 0.01
    assert model.epochs == 1000
    assert model.loss == 'mse'

def test_invalid_loss() -> None:
    """Tests if invalid loss raises ValueError."""
    with pytest.raises(ValueError, match='Unknown loss'):
        LinearRegressionGD(loss='invalid_loss')

def test_fit_convergence(linear_data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    """Tests if model converges."""
    x, y = linear_data
    model = LinearRegressionGD(lr=0.01, epochs=2000, loss='mse')
    model.fit(x, y)
    # Model should nearly have w=2 and b=1.
    assert pytest.approx(model.w, abs=1e-2) == 2.0
    assert pytest.approx(model.b, abs=1e-1) == 1.0
    assert model.score(x, y) > 0.99
