import numpy as np

class LinearRegressionGD:
    """
    LinearRegressionGD: single-class implementation (model + losses + gradients)

    Model (1D input x):
        ŷ = w * x + b

    Losses (for training choices):
      - MSE (mean squared error)
          J(w,b) = (1/m) * Σ_i (y_i - ŷ_i)^2
        gradients:
          ∂J/∂w = (-2/m) Σ_i x_i (y_i - ŷ_i)
          ∂J/∂b = (-2/m) Σ_i (y_i - ŷ_i)

      - MAE (mean absolute error)
          J(w,b) = (1/m) * Σ_i |y_i - ŷ_i|
        subgradients (use sign; not differentiable at 0 — we use sign(·)):
          ∂J/∂w ≈ (-1/m) Σ_i x_i sign(y_i - ŷ_i)
          ∂J/∂b ≈ (-1/m) Σ_i sign(y_i - ŷ_i)

      - Huber (smooth combination)
          for error e = y - ŷ and threshold δ:
            loss = 0.5 * e^2                if |e| <= δ
                 = δ * |e| - 0.5 * δ^2    otherwise
        gradients follow piecewise rules.

    Usage:
      model = LinearRegressionGD(lr=0.01, epochs=1000, loss='mse', verbose=False)
      model.fit(X, y)
      preds = model.predict(X)
      model.score(X, y)  # R^2
    """

    def __init__(self, lr=0.01, epochs=1000, loss='mse', delta=1.0, verbose=False):
        """
        Initialize the LinearRegressionGD model.
        Args:
            lr (float): Learning rate for gradient descent.
            epochs (int): Number of training epochs.
            loss (str): Loss function to use ('mse', 'mae', 'huber').
            delta (float): Delta parameter for Huber loss.
            verbose (bool): If True, prints progress during training.
        """
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.loss = loss.lower()
        self.delta = float(delta)  # for Huber loss
        self.verbose = verbose

        # parameters
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

        # map loss name -> (loss_fn, grad_fn)
        self._loss_map = {
            'mse':   (self._mse, self._mse_grad),
            'mae':   (self._mae, self._mae_grad),
            'huber': (self._huber, self._huber_grad),
        }
        if self.loss not in self._loss_map:
            raise ValueError(f"Unknown loss '{loss}'. choose from {list(self._loss_map.keys())}")

    
    def fit(self, X, y):
        """
        Train the model using batch (vectorized) gradient descent.
        Args:
            X (array-like): Input features (1D array).
            y (array-like): Target values (1D array).
        """
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)
        m = len(y)
        if m == 0:
            raise ValueError("Empty dataset")

        loss_fn, grad_fn = self._loss_map[self.loss]

        # init parameters (kept simple)
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            y_pred = self.w * X + self.b

            # compute gradients
            dw, db = grad_fn(X, y, y_pred)

            # update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # compute and store loss
            loss = loss_fn(y, y_pred)
            self.loss_history.append(loss)

            if self.verbose and (epoch % max(1, (self.epochs // 10)) == 0):
                print(f"[{self.loss.upper()}] epoch={epoch:4d} loss={loss:.6f} w={self.w:.4f} b={self.b:.4f}")

    def predict(self, X):
        """
        Predict target values for given input X.
        Args:
            X (array-like): Input features (1D array).
        Returns:
            np.ndarray: Predicted values.
        """
        X = np.asarray(X).reshape(-1)
        return self.w * X + self.b

    def score(self, X, y):
        """
        Compute the coefficient of determination R^2.
        Args:
            X (array-like): Input features (1D array).
            y (array-like): True target values (1D array).
        Returns:
            float: R^2 score.
        """
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

    def _mse(self, y, y_pred):
        """
        Compute Mean Squared Error (MSE) loss.
        Args:
            y (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: MSE loss.
        """
        return np.mean((y - y_pred)**2)

    def _mse_grad(self, X, y, y_pred):
        """
        Compute gradients of MSE loss with respect to w and b.
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            tuple: (dw, db) gradients.
        """
        m = len(y)
        dw = (-2.0 / m) * np.sum(X * (y - y_pred))
        db = (-2.0 / m) * np.sum(y - y_pred)
        return dw, db

    def _mae(self, y, y_pred):
        """
        Compute Mean Absolute Error (MAE) loss.
        Args:
            y (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: MAE loss.
        """
        return np.mean(np.abs(y - y_pred))

    def _mae_grad(self, X, y, y_pred):
        """
        Compute subgradients of MAE loss with respect to w and b.
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            tuple: (dw, db) subgradients.
        """
        m = len(y)
        sign = np.sign(y - y_pred)   # subgradient sign(y - y_pred)
        dw = (-1.0 / m) * np.sum(X * sign)
        db = (-1.0 / m) * np.sum(sign)
        return dw, db

    def _huber(self, y, y_pred):
        """
        Compute Huber loss.
        Args:
            y (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            float: Huber loss.
        """
        err = y - y_pred
        d = self.delta
        small = np.abs(err) <= d
        small_loss = 0.5 * err[small]**2
        big_loss = d * np.abs(err[~small]) - 0.5 * d**2
        # mean across elements
        # combine using counts to keep vectorization stable (but mean of concat is ok too)
        return (np.sum(small_loss) + np.sum(big_loss)) / len(err)

    def _huber_grad(self, X, y, y_pred):
        """
        Compute gradients of Huber loss with respect to w and b.
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
        Returns:
            tuple: (dw, db) gradients.
        """
        err = y - y_pred
        d = self.delta
        m = len(y)
        # gradient for each sample (∂loss/∂ŷ), note we want gradient wrt parameters: ∂loss/∂w = Σ x_i * ∂loss/∂ŷ_i
        grad_per_sample = np.where(np.abs(err) <= d, -err, -d * np.sign(err))
        dw = (1.0 / m) * np.sum(X * grad_per_sample)
        db = (1.0 / m) * np.sum(grad_per_sample)
        return dw, db