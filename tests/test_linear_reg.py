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

def test_mse_convergence(linear_data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    """Verify MSE loss decreases and finds correct parameters."""
    x, y = linear_data
    model: LinearRegressionGD = LinearRegressionGD(lr=0.01, epochs=500, loss='mse')
    model.fit(x, y)

    assert model.loss_history[-1] < model.loss_history[0]
    assert pytest.approx(model.w, abs=0.1) == 2.0
    assert pytest.approx(model.b, abs=0.1) == 1.0

def test_mae_convergence(linear_data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    """Verify MAE loss decreases and converges."""
    x, y = linear_data
    model: LinearRegressionGD = LinearRegressionGD(lr=0.05, epochs=500, loss='mae')
    model.fit(x, y)

    assert model.loss_history[-1] < model.loss_history[0]
    assert pytest.approx(model.w, abs=0.2) == 2.0

def test_huber_convergence(linear_data: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> None:
    """Verify Huber loss decreases and converges."""
    x, y = linear_data
    model: LinearRegressionGD = LinearRegressionGD(lr=0.02, epochs=500, loss='huber', delta=1.0)
    model.fit(x, y)

    assert model.loss_history[-1] < model.loss_history[0]
    assert pytest.approx(model.w, abs=0.1) == 2.0

def test_r2_score() -> None:
    """Test R^2 score for a perfect fit."""
    x: npt.NDArray[np.float64] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y: npt.NDArray[np.float64] = 2.0 * x + 1.0
    model: LinearRegressionGD = LinearRegressionGD(lr=0.01, epochs=1000)
    model.fit(x, y)
    assert model.score(x, y) > 0.99

def test_fit_empty_dataset_raises() -> None:
    """Test that fitting on an empty dataset raises ValueError."""
    model = LinearRegressionGD()
    with pytest.raises(ValueError, match='Empty dataset'):
        model.fit([], [])


def test_r2_score_zero_variance_targets() -> None:
    """Test R^2 score when target values have zero variance."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 5.0, 5.0])  # zero variance

    model = LinearRegressionGD()
    model.fit(x, y)

    assert model.score(x, y) == 0.0
