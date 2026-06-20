# Calculus Essentials

> **Course:** Data Science Foundations
> **Lesson:** Day 6 - Math for Data Science II (Calculus)
> **Prerequisites:** Statistics Fundamentals, Distributions, Probability, Hypothesis Testing, Linear Algebra Essentials


## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what a derivative is, both geometrically and intuitively, and compute derivatives using the power, sum, product, and chain rules
- Interpret what the sign and magnitude of a derivative tell you about a function
- Compute partial derivatives of functions of several variables
- Assemble partial derivatives into the **gradient vector**, and explain why it points in the direction of steepest ascent
- Implement **gradient descent** from scratch and apply it to fit a linear regression model
- Connect this iterative approach back to the closed-form normal equations from the linear algebra's lesson


## 1. Why Calculus Matters for Data Science

Previously you solved linear regression with a single elegant formula: $\vec{w} = (X^T X)^{-1} X^T \vec{y}$. The normal equations give you the optimal weights in one shot, no iteration required. So why bother with calculus at all?

The honest answer is that *most* models in data science do not have closed-form solutions. Logistic regression, neural networks, decision trees, gradient-boosted models, and essentially every modern deep learning architecture share a single principle for finding their parameters:

> *"Define a loss function. Compute its derivative with respect to the parameters. Adjust the parameters in the direction that reduces the loss. Repeat."*

That entire recipe is calculus in action. This lesson teaches you what a derivative is, what a gradient is, and how to use both to make a model learn from data. By the end you will have implemented the engine, **gradient descent**, that powers nearly all modern machine learning.

## 2. The Derivative

### 2.1 Intuition: The Slope of a Curve

A derivative measures **how fast a function is changing** at a particular point. Visually, imagine a function $f(x)$ drawn as a curve. At any specific point on that curve, you can draw a tangent line, a line that just kisses the curve at that point. The **slope of that tangent line** is the derivative of $f$ at that point.

```
     y = f(x)
       |
       |        .---.
       |      /       \         <-- tangent line here is flat: slope = 0
       |    /           \           (this is a maximum)
       |  /               \
       |/                   \
       +-------- x ----------\---
                              \
                               \   <-- tangent line here slopes downward: slope < 0
                                \      (function is decreasing)
```

The derivative is written:

$$f'(x) \qquad \text{or} \qquad \frac{df}{dx}$$

Both notations mean the same thing. The first is shorthand; the second emphasises that the derivative is a *ratio*, a tiny change in $f$ divided by a tiny change in $x$.

### 2.2 The Formal Definition

The slope of a line through two points $(x, f(x))$ and $(x+h, f(x+h))$ is the familiar rise-over-run:

$$\text{slope} = \frac{f(x+h) - f(x)}{h}$$

The derivative is what this slope becomes as $h$ shrinks to zero, that is, as the two points squeeze together into a single tangent point:

$$\boxed{f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}}$$

You will almost never compute derivatives from this limit definition in practice. Instead, you will use a handful of rules that the limit definition produces, summarised next.

### 2.3 The Essential Derivative Rules

| Rule | Function | Derivative | Example |
|---|---|---|---|
| **Constant** | $f(x) = c$ | $f'(x) = 0$ | $\frac{d}{dx}(7) = 0$ |
| **Power** | $f(x) = x^n$ | $f'(x) = n x^{n-1}$ | $\frac{d}{dx}(x^3) = 3x^2$ |
| **Constant multiple** | $f(x) = c \cdot g(x)$ | $f'(x) = c \cdot g'(x)$ | $\frac{d}{dx}(5x^2) = 10x$ |
| **Sum** | $f(x) = g(x) + h(x)$ | $f'(x) = g'(x) + h'(x)$ | $\frac{d}{dx}(x^2 + 3x) = 2x + 3$ |
| **Product** | $f(x) = g(x) \cdot h(x)$ | $f'(x) = g'(x) h(x) + g(x) h'(x)$ | $\frac{d}{dx}(x^2 \cdot \sin x) = 2x \sin x + x^2 \cos x$ |
| **Chain** | $f(x) = g(h(x))$ | $f'(x) = g'(h(x)) \cdot h'(x)$ | $\frac{d}{dx}((3x + 1)^2) = 2(3x+1) \cdot 3$ |

### 2.4 Worked Examples

**Example 1.** Differentiate $f(x) = x^2$.

By the power rule, $f'(x) = 2x$. Sanity check: at $x = 0$, the slope is 0 (the parabola is flat at the bottom). At $x = 1$, the slope is 2. At $x = 2$, the slope is 4. The function is getting steeper as we move right, which matches what a parabola looks like.

**Example 2.** Differentiate $f(x) = 3x^4 - 5x^2 + 7$.

Apply the sum and constant-multiple rules term by term:

$$f'(x) = 12x^3 - 10x + 0 = 12x^3 - 10x$$

**Example 3.** Differentiate $f(x) = (x^2 + 1)^3$ using the chain rule.

Identify the outer function $g(u) = u^3$ and the inner function $h(x) = x^2 + 1$. Then $g'(u) = 3u^2$ and $h'(x) = 2x$, so:

$$f'(x) = 3(x^2 + 1)^2 \cdot 2x = 6x(x^2 + 1)^2$$

> **Why the chain rule is the most important rule in machine learning:** every deep neural network is just a long composition of simple functions. To train it, we need the derivative of the loss with respect to every parameter, which means applying the chain rule one layer at a time. This is exactly what **backpropagation** does. If you only deeply learn one differentiation rule, learn this one.

### 2.5 What Does the Sign of the Derivative Tell Us?

This is the practical payoff that makes derivatives so useful in optimisation:

| If... | Then... | Meaning |
|---|---|---|
| $f'(x) > 0$ | $f$ is **increasing** at $x$ | Moving right makes $f$ go up |
| $f'(x) < 0$ | $f$ is **decreasing** at $x$ | Moving right makes $f$ go down |
| $f'(x) = 0$ | $x$ is a **critical point** | A local max, local min, or saddle point |

To **minimise** a function, we want to find a point where $f'(x) = 0$. To **iteratively** approach that point, we want to step in the direction *opposite* the derivative: if the derivative is positive, go left; if negative, go right. That single observation is the seed of gradient descent.


## 3. Partial Derivatives

### 3.1 When Functions Have Several Inputs

Real machine learning loss functions depend on many parameters at once, often millions. A function of two variables might look like:

$$f(x, y) = x^2 + 3xy + y^2$$

This is a surface in 3D, not a curve in 2D:

```
     z = f(x, y)
        ___________
       /  ___       \
      / /     \      \         <-- a surface, like a bowl or saddle
     | |       |      |
      \ \_____/      /
       \___________/
                y
              /
            /
        +----- x
```

How do we measure "how fast $f$ is changing" on a surface? It depends on which direction you walk in. The **partial derivative** measures how fast $f$ changes if you walk in exactly one coordinate direction while keeping all others fixed.

### 3.2 Notation

The partial derivative of $f$ with respect to $x$, holding $y$ fixed, is written:

$$\frac{\partial f}{\partial x}$$

(That curly $\partial$ symbol is the cue: this is a partial derivative, not an ordinary one.) Similarly $\frac{\partial f}{\partial y}$ holds $x$ fixed.

### 3.3 How to Compute Them

This is the most reassuring fact in all of multivariable calculus: a partial derivative is computed by **treating all other variables as constants**, then differentiating normally.

**Worked example.** Let $f(x, y) = x^2 + 3xy + y^2$.

To compute $\frac{\partial f}{\partial x}$, treat $y$ as a constant:

$$\frac{\partial f}{\partial x} = 2x + 3y + 0 = 2x + 3y$$

To compute $\frac{\partial f}{\partial y}$, treat $x$ as a constant:

$$\frac{\partial f}{\partial y} = 0 + 3x + 2y = 3x + 2y$$

That is the entire skill. No new rules; just the existing ones applied one variable at a time.

### 3.4 A Second Example

Let $f(w_1, w_2, w_3) = (w_1 + 2 w_2 + 3 w_3 - 10)^2$. This has the shape of a loss function (a squared error). Using the chain rule:

$$\frac{\partial f}{\partial w_1} = 2(w_1 + 2 w_2 + 3 w_3 - 10) \cdot 1 = 2(w_1 + 2 w_2 + 3 w_3 - 10)$$

$$\frac{\partial f}{\partial w_2} = 2(w_1 + 2 w_2 + 3 w_3 - 10) \cdot 2 = 4(w_1 + 2 w_2 + 3 w_3 - 10)$$

$$\frac{\partial f}{\partial w_3} = 2(w_1 + 2 w_2 + 3 w_3 - 10) \cdot 3 = 6(w_1 + 2 w_2 + 3 w_3 - 10)$$

This kind of pattern appears everywhere in linear regression. We will see exactly this in Section 5.


## 4. The Gradient

### 4.1 Definition

For a function $f(\vec{w})$ that takes a vector of $n$ inputs, the **gradient** is the vector containing all $n$ partial derivatives, in order:

$$\nabla f(\vec{w}) = \begin{bmatrix} \dfrac{\partial f}{\partial w_1} \\ \dfrac{\partial f}{\partial w_2} \\ \vdots \\ \dfrac{\partial f}{\partial w_n} \end{bmatrix}$$

The symbol $\nabla$ (a triangle pointing down, called "delta" ) simply means "gradient of."

### 4.2 What the Gradient Tells You Geometrically

Here is the result that makes gradient descent work. At any point $\vec{w}$, the gradient $\nabla f(\vec{w})$ is a vector with two important properties:

1. **It points in the direction of steepest ascent.** No other direction increases $f$ faster.
2. **Its magnitude tells you the slope** in that steepest direction.

Picture standing on a hillside in fog. You cannot see the summit, but you can feel which direction is steepest uphill. That direction, at your current position, is exactly the gradient.

```
            ^  
            |    direction of steepest ascent = +grad f
            |
       ---- * ----   <-- your current position
            |
            v
              direction of steepest descent = -grad f
```

### 4.3 Why This Matters for Data Science

Machine learning is about *minimising* a loss function (mean squared error, cross-entropy, etc.). If we know the direction of steepest **ascent**, then by symmetry the direction of steepest **descent** is exactly the opposite: $-\nabla f(\vec{w})$.

That single observation is the engine of essentially all modern ML training.


## 5. Gradient Descent

### 5.1 The Big Idea

To minimise $f(\vec{w})$:

1. Start at some initial guess $\vec{w}_0$.
2. Compute the gradient $\nabla f(\vec{w}_0)$.
3. Take a small step in the **opposite** direction: $\vec{w}_1 = \vec{w}_0 - \alpha \nabla f(\vec{w}_0)$.
4. Repeat from the new point.

The scalar $\alpha$ is called the **learning rate**. It controls how big each step is.

```
   f(w)
    |          (start)
    |         *
    |          \
    |           \
    |            *
    |             \
    |              *
    |               \  
    |                * --- *  (converged near the minimum)
    +----------------------------- w
```

Each arrow is one update step. The path may zig-zag a little, but as long as the learning rate is reasonable, the iterations will roll downhill toward a minimum.

### 5.2 The Algorithm in One Line

$$\boxed{\vec{w}_{t+1} = \vec{w}_t - \alpha \nabla f(\vec{w}_t)}$$

Repeat until either the gradient becomes very small (we are near a critical point) or we hit a fixed iteration budget.

### 5.3 Choosing the Learning Rate

The learning rate $\alpha$ is the most important hyperparameter in gradient descent:

| Learning rate | What happens |
|---|---|
| Too small | Convergence is very slow; you may run out of iterations before getting close to the minimum |
| Just right | Smooth, steady decrease in loss; converges efficiently |
| Too large | Overshoots the minimum each step; loss oscillates or even *increases* (divergence) |

There is no universal "best" learning rate; it depends on the function being optimised and the scale of the features. This is one reason **feature scaling** (which you applied yesterday) is so helpful: it keeps the gradient components on a comparable scale, making it much easier to find a single $\alpha$ that works.

### 5.4 Gradient Descent for Linear Regression

This is the moment everything we have built this week converges. Recall the linear regression loss from yesterday:

$$L(\vec{w}) = \frac{1}{m} \|\vec{y} - X\vec{w}\|^2 = \frac{1}{m} \sum_{i=1}^{m}(y_i - X_i \vec{w})^2$$

Taking partial derivatives and assembling the gradient (using the same pattern from Section 3.4) gives:

$$\nabla L(\vec{w}) = -\frac{2}{m} X^T (\vec{y} - X\vec{w})$$

So the gradient descent update for linear regression is:

$$\vec{w}_{t+1} = \vec{w}_t + \frac{2 \alpha}{m} X^T (\vec{y} - X\vec{w}_t)$$

This is exactly the formula we will implement in code below.

### 5.5 Normal Equations vs Gradient Descent

Both methods solve the same least-squares problem. When should you reach for each?

| | Normal Equations | Gradient Descent |
|---|---|---|
| Solution style | Closed-form, one step | Iterative, many steps |
| Cost per iteration | None; one matrix inversion total | One pass through the data |
| Scales to many features | Expensive: inverting $X^T X$ is $O(n^3)$ | Excellent: each step is $O(mn)$ |
| Scales to many samples | OK | Excellent, especially with stochastic variants |
| Requires a learning rate | No | Yes (must be tuned) |
| Works for non-linear models | No (only linear regression) | Yes (powers nearly all modern ML) |

For linear regression with a few hundred features, the normal equations win on simplicity. For modern deep learning models with billions of parameters, gradient descent (and its many descendants: SGD, Adam, RMSProp, etc.) is the only option that scales.


## 6. Hands-On: Calculus and Gradient Descent on California Housing

We will use the same **California Housing dataset** as yesterday (20,640 samples, 8 features), which lets us directly compare today's gradient descent solution against yesterday's normal equations solution on identical data.

### 6.1 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

sns.set(style="whitegrid")
np.random.seed(42)

# Load and prepare the dataset (same as Day 5)
data = fetch_california_housing()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
y_full = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full.values)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=42
)

# Add intercept column (same as Day 5)
def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_train_d = add_intercept(X_train)
X_test_d = add_intercept(X_test)

print(f"Training samples: {X_train_d.shape[0]:,}")
print(f"Features (incl. intercept): {X_train_d.shape[1]}")
```

### 6.2 Numerical Derivatives: Approximating $f'(x)$

Before we run gradient descent on the whole dataset, let us confirm that the rules from Section 2 work by approximating derivatives numerically. The idea is straightforward: use a tiny $h$ in the rise-over-run formula.

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Check against the power rule: d/dx (x^3) should be 3x^2
f = lambda x: x ** 3
exact_at_2 = 3 * (2 ** 2)
numerical_at_2 = numerical_derivative(f, 2.0)

print(f"Exact derivative at x=2:       {exact_at_2}")
print(f"Numerical derivative at x=2:   {numerical_at_2:.6f}")

# Visualise f and f' together
x_vals = np.linspace(-3, 3, 200)
f_vals = f(x_vals)
fprime_vals = np.array([numerical_derivative(f, x) for x in x_vals])

plt.figure(figsize=(9, 4))
plt.plot(x_vals, f_vals, label="f(x) = x^3", color="steelblue", lw=2)
plt.plot(x_vals, fprime_vals, label="f'(x) (numerical)", color="tomato", lw=2)
plt.axhline(0, color="black", lw=0.5)
plt.legend()
plt.title("A function and its derivative")
plt.show()
```

### 6.3 Visualising a Loss Surface (Two-Feature Subproblem)

To see what gradient descent is actually doing, we will use just **two features** from California Housing so we can plot the full loss surface in 3D.

```python
# Use just MedInc (column 0) and HouseAge (column 1) for visualisation
X_two = X_train[:, :2]
y_two = y_train

def mse_loss(w0, w1):
    preds = X_two[:, 0] * w0 + X_two[:, 1] * w1
    return np.mean((y_two - preds) ** 2)

# Build a grid of (w0, w1) values and compute the loss at each
w0_grid = np.linspace(-2, 4, 80)
w1_grid = np.linspace(-2, 2, 80)
W0, W1 = np.meshgrid(w0_grid, w1_grid)
L = np.zeros_like(W0)
for i in range(W0.shape[0]):
    for j in range(W0.shape[1]):
        L[i, j] = mse_loss(W0[i, j], W1[i, j])

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(W0, W1, L, cmap="viridis", alpha=0.85)
ax1.set_xlabel("w0 (MedInc)")
ax1.set_ylabel("w1 (HouseAge)")
ax1.set_zlabel("Loss (MSE)")
ax1.set_title("Loss surface in 3D")

ax2 = fig.add_subplot(1, 2, 2)
cs = ax2.contour(W0, W1, L, levels=25, cmap="viridis")
ax2.clabel(cs, inline=True, fontsize=8)
ax2.set_xlabel("w0 (MedInc)")
ax2.set_ylabel("w1 (HouseAge)")
ax2.set_title("Contour view (top-down)")

plt.tight_layout()
plt.show()
```

The bowl shape is the visual signature of a **convex** loss function. Gradient descent on a convex bowl is guaranteed to find the bottom, regardless of where you start.

### 6.4 Implementing Gradient Descent for Linear Regression

Now the real thing. We will implement the update rule from Section 5.4 and run it on the full 8-feature California Housing dataset.

```python
def linear_regression_gd(X, y, learning_rate=0.01, n_iters=1000, verbose_every=100):
    """
    Linear regression by gradient descent.

    Parameters
    ----------
    X : (m, n) design matrix (intercept column expected)
    y : (m,) target vector
    learning_rate : alpha
    n_iters : number of update steps

    Returns
    -------
    w : (n,) learned weights
    loss_history : list of MSE values at each iteration
    """
    m, n = X.shape
    w = np.zeros(n)
    loss_history = []

    for t in range(n_iters):
        # Predictions and residuals
        y_pred = X @ w
        residuals = y - y_pred

        # Loss (MSE)
        loss = np.mean(residuals ** 2)
        loss_history.append(loss)

        # Gradient of MSE w.r.t. w (Section 5.4)
        grad = -(2 / m) * (X.T @ residuals)

        # Update step
        w = w - learning_rate * grad

        if (t + 1) % verbose_every == 0:
            print(f"Iter {t+1:4d}  loss = {loss:.6f}")

    return w, loss_history

w_gd, losses = linear_regression_gd(X_train_d, y_train,
                                     learning_rate=0.05, n_iters=1000,
                                     verbose_every=200)
```

### 6.5 Tracking the Loss Curve

```python
plt.figure(figsize=(9, 4))
plt.plot(losses, color="steelblue", lw=2)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Loss Curve (California Housing, 8 features)")
plt.grid(True, alpha=0.3)
plt.show()

print(f"Initial loss:  {losses[0]:.4f}")
print(f"Final loss:    {losses[-1]:.4f}")
print(f"Reduction:     {(1 - losses[-1] / losses[0]) * 100:.2f}%")
```

A smooth, monotonically decreasing curve is the signature of healthy convergence. If you see oscillations or an increasing curve, your learning rate is too high.

### 6.6 Comparing Gradient Descent to the Normal Equations

```python
# Re-compute the normal equations solution for direct comparison
w_normal = np.linalg.inv(X_train_d.T @ X_train_d) @ X_train_d.T @ y_train

feature_names = ["Intercept"] + list(X_full.columns)
comparison = pd.DataFrame({
    "Feature": feature_names,
    "Normal Equations": w_normal,
    "Gradient Descent": w_gd,
    "Abs. Difference": np.abs(w_normal - w_gd)
})
print(comparison.round(6))
```

The two columns should be nearly identical (typically agreeing to 4 or 5 decimal places). Gradient descent gets to the same answer as the normal equations, just by a different route. This is your evidence that the calculus underneath is sound.

```python
# Compare predictions on the test set
y_pred_gd = X_test_d @ w_gd
y_pred_normal = X_test_d @ w_normal

print(f"R^2 (Gradient Descent):  {r2_score(y_test, y_pred_gd):.4f}")
print(f"R^2 (Normal Equations):  {r2_score(y_test, y_pred_normal):.4f}")
```

### 6.7 The Learning Rate Matters: A Side-by-Side Comparison

To see the practical impact of $\alpha$, let us run gradient descent with three different learning rates and overlay the loss curves.

```python
lr_too_small = 0.0001
lr_good = 0.05
lr_too_large = 0.5

_, losses_small = linear_regression_gd(X_train_d, y_train,
                                       learning_rate=lr_too_small,
                                       n_iters=1000, verbose_every=10_000)
_, losses_good = linear_regression_gd(X_train_d, y_train,
                                      learning_rate=lr_good,
                                      n_iters=1000, verbose_every=10_000)
_, losses_large = linear_regression_gd(X_train_d, y_train,
                                       learning_rate=lr_too_large,
                                       n_iters=1000, verbose_every=10_000)

plt.figure(figsize=(10, 5))
plt.plot(losses_small, label=f"Too small (alpha={lr_too_small})", color="steelblue", lw=2)
plt.plot(losses_good, label=f"Just right (alpha={lr_good})", color="seagreen", lw=2)
plt.plot(losses_large, label=f"Too large (alpha={lr_too_large})", color="tomato", lw=2)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("How the Learning Rate Affects Convergence")
plt.legend()
plt.yscale("log")
plt.show()
```

You should see three distinct behaviours: the small learning rate barely moves, the good one converges quickly, and the large one either oscillates or explodes upward. There is no skipping this step in practice; tuning $\alpha$ is part of the job.

### 6.8 Visualising Gradient Descent in 2D

To close the loop on Section 6.3, let us run gradient descent on the two-feature subproblem and overlay the path on the contour plot.

```python
def gd_path(X, y, lr=0.1, n_iters=50, start=(-1.5, 1.5)):
    w = np.array(start, dtype=float)
    path = [w.copy()]
    m = len(y)
    for _ in range(n_iters):
        residual = y - X @ w
        grad = -(2 / m) * (X.T @ residual)
        w = w - lr * grad
        path.append(w.copy())
    return np.array(path)

path = gd_path(X_two, y_two, lr=0.1, n_iters=50, start=(-1.5, 1.5))

plt.figure(figsize=(8, 6))
cs = plt.contour(W0, W1, L, levels=25, cmap="viridis")
plt.clabel(cs, inline=True, fontsize=8)
plt.plot(path[:, 0], path[:, 1], "o-", color="tomato", markersize=5,
         label="Gradient descent path")
plt.scatter([path[0, 0]], [path[0, 1]], color="red", s=120, marker="*",
            label="Start")
plt.scatter([path[-1, 0]], [path[-1, 1]], color="black", s=120, marker="X",
            label="End")
plt.xlabel("w0 (MedInc)")
plt.ylabel("w1 (HouseAge)")
plt.title("Gradient Descent Path on the Loss Surface")
plt.legend()
plt.show()
```

You should see the path step downhill, perpendicular to the contour lines at every point (because the gradient itself is always perpendicular to the contours). That is the geometric meaning of "steepest descent."

## Daily Challenge

1. **Derivatives by hand.** Compute, by hand, the derivatives of: (a) $f(x) = 4x^5 - 2x^3 + 7x - 1$, (b) $f(x) = (2x + 3)^4$, (c) $f(x) = x^2 \sin(x)$. Verify each numerically using the `numerical_derivative` function from Section 6.2.

2. **Partial derivatives by hand.** For $f(w_1, w_2) = w_1^2 + 4 w_1 w_2 + 3 w_2^2$: (a) compute $\frac{\partial f}{\partial w_1}$ and $\frac{\partial f}{\partial w_2}$, (b) write out the gradient vector $\nabla f$, (c) evaluate the gradient at the point $(1, 2)$.

3. **Gradient descent from scratch.** Apply the gradient descent function from Section 6.4 to the **Diabetes dataset** (`sklearn.datasets.load_diabetes`). Standardise the features, split into train and test, and compare the resulting weights to a normal-equations solution. Report the test R² for both methods.

4. **Tuning the learning rate.** Run gradient descent on the California Housing data with at least 5 different learning rates spanning several orders of magnitude (e.g. 1e-5, 1e-3, 0.01, 0.1, 1.0, 10). For each, report the final loss and whether it converged. Plot all loss curves on a single figure with a log scale.

5. **Bonus.** The gradient descent path in Section 6.8 should be perpendicular to the contour lines at every step. Explain in 2 to 3 sentences why this is the case, using the geometric meaning of the gradient from Section 4.2.


## Summary

| Concept | Notation | What it represents |
|---|---|---|
| Derivative | $f'(x)$ or $\frac{df}{dx}$ | Instantaneous rate of change; slope of tangent line |
| Power rule | $\frac{d}{dx} x^n = n x^{n-1}$ | Differentiating monomials |
| Chain rule | $\frac{d}{dx} g(h(x)) = g'(h(x)) \cdot h'(x)$ | Differentiating compositions; the backbone of backpropagation |
| Partial derivative | $\frac{\partial f}{\partial w_i}$ | Rate of change when only $w_i$ varies |
| Gradient | $\nabla f$ | Vector of all partial derivatives; points in direction of steepest ascent |
| Gradient descent update | $\vec{w}_{t+1} = \vec{w}_t - \alpha \nabla f(\vec{w}_t)$ | Iterative minimisation algorithm |
| Learning rate | $\alpha$ | Controls step size; must be tuned |
| MSE gradient (linear regression) | $\nabla L = -\frac{2}{m} X^T (\vec{y} - X\vec{w})$ | Used inside the GD loop |



## Module Wrap-Up: The Full Statistics and Math Journey

You have now completed the entire Statistics and Math for Data Science module. Here is how every day connects:

```
Mon  ->  Stats Fundamentals       Describe data
Tue  ->  Distributions             Model the data
Wed  ->  Probability Theory        Quantify uncertainty
Thu  ->  Hypothesis Testing        Make decisions from data
Fri  ->  Linear Algebra            Structure the data and the model
Sat  ->  Calculus                  Optimise the model from data
```

Each piece supports the next. Statistics tells you what your data looks like. Distributions give you a mathematical model for that shape. Probability lets you reason under uncertainty. Hypothesis testing turns that reasoning into decisions. Linear algebra gives you the language for working with multi-feature data and multi-parameter models. And calculus gives you the engine that makes those models *learn*.

Every machine learning algorithm you study from this point forward, whether it is logistic regression, k-means, neural networks, or gradient-boosted trees, is built from exactly these ingredients. You now have the foundation to read any standard reference, follow any derivation, and implement any standard algorithm from first principles.


*Lesson designed by Zindua School Team members, co-authored by Claude and Counterchecked by Zindua School Data Science Team*
