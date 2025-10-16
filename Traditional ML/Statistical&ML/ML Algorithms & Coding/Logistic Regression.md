# Logistic Regression

## Theory

### Math

Sigmoid activation
$$
h(x)=\frac{1}{1+e^{-z}}=\frac{1}{1+e^{-wX+b}}
$$
The goal is to find weights w and bias b that **maximize** the likelihood of observing the data.

Probability
$$
p(y|x;\theta)=h_\theta(x)^y(1-h_\theta(x))^{(1-y)}
$$
Likelihood of Parameters
$$
L(\theta)=\Pi^m_{i=1}h_\theta(x^{(i)})^{y^{(i)}}(1-h_\theta(x)^{(i)})^{(1-y^{(i)})}
$$
Log likelihood
$$
l(\theta)=\sum^m_{i=1}y^{(i)}\log h_\theta(x^{(i)})+(1-y^{(i)})\log (1-h_\theta(x^{(i)}))
$$
### Gradient

To find the best w and b we use **gradient ascent** on the **log-likelihood function** - the $l(\theta)$
$$
l'_{w_j}=\frac{1}{n}\sum_{i=1}^n(y_i-h_\theta(x))x_{ij}
$$

$$
\theta_j=\theta_j+\alpha l'_{w_j}\\=\theta_j+\alpha\frac{1}{n}\sum^n_{i=1}x^{(i)}(y^{(i)}-h_\theta(x^{(i)}))
$$

### Loss

That gives the loss to minimize - **negative log-likelihood**

## Pseudo Code

```python
def LogisticRegression(features, labels, weights, lr, epoch):
    for i in epoch:
        weights = updateWeights(features, labels, weights, lr)
    return weights

def sigmoid(x):
    return 1 / (1 + exp(-x))

def h(features, weights):
    return sigmoid(np.dot(features, weights))

def updateWeights(features, labels, weights, lr):
    # features: [x, f]
    # labels: [x]
    # weights: [f]
    predictions = h(features, weights)  # [x, f] * [f] -> [x]
    gradient = np.dot(features.T, labels-predictions)  # [f, x] * [x] -> [f]
    gradient /= labels.size
    weights += lr * gradient
    return weights
```

## Coding

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # prediction
            y_predicted = self.sigmoid(linear_model)

            # gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        return np.where(self.predict_proba(X) >= threshold, 1, 0)
```

