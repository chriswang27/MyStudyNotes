# Linear Regression

## Theory

### Variance-Bias Tradeoff

Low MSE = Low Variance + Low Bias

- Variance: the amount by which $\hat{f}$ would change if we used different training data.
- Bias is the error introduce by modeling a real-life problem with a simpler model.

Generally, as flexibility increases, variance will increase and bias will decrease.

### Loss

In Linear Regression, the Mean Squared Error (MSE) cost function is employed.
$$
J(w,b)=\frac{1}{n}\sum^n_{i=1}(\hat y_i - y_i)^2
$$


### Gradient

$$
J'_{w_{ij}}=\frac{1}{n}\sum^n_{i=1}(\hat y_i-y)x_{ij}\\
J'_{b_i}=\frac{1}{n}\sum^n_{i=1}(\hat y_i-y)
$$
### Update

$$
w_{ij}=w_{ij}-\alpha*J'_{w_{ij}}\\
b_{i}=b_{i}-\alpha*J'_{b_{i}}
$$

## Pseudo Code

```python
LinearRegression(epoch, learning_rate, X, Y, feature_size, epsilon):
    thetas[feature_size + 1] = 0
    for i = 1 to epoch:
        grads[feature_size + 1] = 0
        # Calculate gradient
        for j = 1 to X.size:
            for k = 1 to featuer_size:
                grads[k] += X[j][k] * (Y[j] - Hypo(thetas, X[j], feature_size))
            grads[k + 1] += Y[j] - Hypo(thetas, X[j], feature_size)
        # Update parameters 
        for k = 1 to feature_size + 1:
            thetas[k] += learning_rate * grads[k] / X.size
        if Norm(grads) < epsilon:
            break
    return thetas
```

```python
Hypo(thetas, x, feature_size):
    h = 0
    for i in feature_size:
        h += thetas[i] * x[i]
    return h
```

```python
Norm(x):
    n = 0
    for i = 1 to x.size:
        n += x[i] * x[i]
    return sqrt(n)
```

