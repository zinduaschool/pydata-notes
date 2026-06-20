# Linear Algebra Essentials

> **Course:** Data Science Foundations
> **Lesson:** Day 5 - Math for Data Science I (Linear Algebra)
> **Prerequisites:** Statistics Fundamentals, Distributions, Probability, Hypothesis Testing, basic Python and NumPy

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what a vector is, both geometrically and algebraically, and perform vector addition, scalar multiplication, and the dot product
- Compute the magnitude (norm) of a vector and interpret what it represents
- Recognise different types of matrices and perform addition, transpose, and scalar multiplication
- Carry out matrix multiplication step-by-step by hand and explain why the rule works the way it does
- Derive and apply the **normal equations** for linear regression, the closed-form solution that powers least squares
- Apply all of the above in Python on a real, large dataset


## 1. Why Linear Algebra Matters for Data Science

So far this week you have treated a dataset as a *collection of measurements*. Today you will start treating it as a **matrix**, and individual observations as **vectors**. That shift is not just notation; it is the entire reason linear algebra is the language of modern data science.

Every machine learning algorithm you will ever meet, from linear regression to deep neural networks, ultimately reduces to:

> *"Take a matrix of data, multiply it by another matrix of parameters, compare the result to what we expected, and adjust."*

Today's lesson teaches you exactly what those operations mean, what they look like, and how to perform them. Calculus will then teach you *how the parameters get adjusted*.


## 2. Vectors: The Building Blocks

### 2.1 What Is a Vector?

A **vector** is, at its simplest, an ordered list of numbers. But it has two equally important interpretations, and a good data scientist holds both in mind at the same time.

**Geometric view:** A vector is an arrow with a length and a direction, drawn from the origin to a point in space.

```
        y
        |
      3 |           * (2, 3)
        |          /
      2 |         /
        |        /
      1 |       /
        |      /
        |     /
        +------------------- x
        0   1   2   3
```

The arrow above represents the 2D vector $\vec{v} = \begin{bmatrix} 2 \\ 3 \end{bmatrix}$.

**Algebraic view:** A vector is just a list of $n$ numbers, where $n$ is called the **dimension**. In code, it is an array. In data science, a single row of a dataset (one customer, one image, one bolt) is a vector. A customer described by 5 features (age, income, tenure, spend, sessions) is a 5-dimensional vector. Geometry stops being easy to draw beyond 3 dimensions, but the algebra works identically in any number of dimensions.

### 2.2 Notation

By convention, vectors are written as **columns**:

$$\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

The individual numbers $v_1, v_2, \ldots$ are called the **components** or **entries**. We write $\vec{v} \in \mathbb{R}^n$ to mean "$\vec{v}$ is a real-valued vector with $n$ components."

### 2.3 Vector Addition

Adding two vectors of the same dimension means adding their components position-by-position:

$$\begin{bmatrix} 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 1 \end{bmatrix} = \begin{bmatrix} 6 \\ 4 \end{bmatrix}$$

**Geometric intuition:** place the tail of the second arrow at the head of the first arrow. The sum is the arrow from the very start to the very end. This is called the **head-to-tail rule** (also known as the parallelogram rule).

```
         y
       4 |               * sum (6, 4)
       3 |     * a (2,3) /|
       2 |       \      / |
       1 |        \    /  | b (4,1)
         |         \  /   |
         +----------\/----+----- x
         0   1   2  3  4  5  6
```

### 2.4 Scalar Multiplication

Multiplying a vector by a regular number (a **scalar**) scales every component:

$$2 \cdot \begin{bmatrix} 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}$$

**Geometric intuition:** the vector gets stretched (if the scalar is greater than 1), shrunk (if between 0 and 1), or flipped to point the opposite way (if negative). The direction either stays the same or reverses; only the length changes.

### 2.5 The Dot Product

The **dot product** is the single most important vector operation in data science. It takes two vectors of the same dimension and produces *one number*:

$$\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**Worked example:**

$$\begin{bmatrix} 2 \\ 3 \end{bmatrix} \cdot \begin{bmatrix} 4 \\ 1 \end{bmatrix} = (2)(4) + (3)(1) = 8 + 3 = 11$$

**Why this matters so much:** the dot product measures **similarity** or **alignment** between two vectors. There is a beautiful geometric formula that makes this explicit:

$$\vec{a} \cdot \vec{b} = \|\vec{a}\| \cdot \|\vec{b}\| \cdot \cos(\theta)$$

where $\theta$ is the angle between the two vectors and $\|\vec{a}\|$ is the length of $\vec{a}$.

| Angle $\theta$ | $\cos(\theta)$ | What it means |
|---|---|---|
| $0°$ (same direction) | $1$ | Vectors point the same way (maximum positive dot product) |
| $90°$ (perpendicular) | $0$ | Vectors are **orthogonal**, no relationship |
| $180°$ (opposite) | $-1$ | Vectors point opposite ways (maximum negative dot product) |

You will use the dot product constantly: cosine similarity between documents, predictions in linear regression (literally $\hat{y} = \vec{w} \cdot \vec{x}$), neuron activations in deep learning, and much more.

### 2.6 Vector Magnitude (Norm)

The **magnitude** or **length** of a vector, written $\|\vec{v}\|$, is computed using the Pythagorean theorem extended to $n$ dimensions:

$$\|\vec{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$

**Worked example:**

$$\left\|\begin{bmatrix} 3 \\ 4 \end{bmatrix}\right\| = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5$$

This is the **Euclidean norm**, also called the **L2 norm**, and it is so common that "norm" usually means this by default.

> **Connection to statistics:** the standard deviation from Monday is essentially the L2 norm of the centred deviations, divided by $\sqrt{n}$. The Euclidean distance you would use to compare two customer profiles is just $\|\vec{a} - \vec{b}\|$. These are not different ideas; they are the same idea wearing different hats.


## 3. Matrices: Organising Data and Transformations

### 3.1 What Is a Matrix?

A **matrix** is a rectangular grid of numbers arranged into rows and columns. In data science, a matrix is almost always a *dataset*: each row is one observation (a vector), and each column is one feature.

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

This is a $3 \times 3$ matrix (3 rows, 3 columns). The convention is **rows by columns**, always in that order.

In code, an entire dataset with $m$ samples and $n$ features is stored as an $m \times n$ matrix. The Iris dataset you used earlier this week is a $150 \times 4$ matrix; the California Housing dataset you will use today is a $20640 \times 8$ matrix.

### 3.2 Notation and Element Access

We refer to the entry in row $i$, column $j$ as $A_{ij}$. In the matrix above, $A_{2,3} = 6$ (row 2, column 3).

### 3.3 Special Matrices Worth Recognising

**Zero matrix** ($\mathbf{0}$): every entry is 0.

**Square matrix:** number of rows equals number of columns.

**Diagonal matrix:** zero everywhere except on the main diagonal:

$$D = \begin{bmatrix} 3 & 0 & 0 \\ 0 & 7 & 0 \\ 0 & 0 & 2 \end{bmatrix}$$

**Identity matrix** ($I$): a diagonal matrix with 1s on the diagonal. It is the multiplicative identity for matrices, meaning $A \cdot I = A$, the matrix equivalent of multiplying a number by 1:

$$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Symmetric matrix:** $A^T = A$. A symmetric matrix mirrors itself across the main diagonal. Covariance and correlation matrices from Monday were both symmetric, and that was not a coincidence.

### 3.4 Transpose

The **transpose** of a matrix, written $A^T$, flips its rows and columns. The first row becomes the first column, the second row becomes the second column, and so on:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \Rightarrow \quad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

A $2 \times 3$ matrix becomes a $3 \times 2$ matrix after transposing. Transposes are everywhere in the normal equations you will derive in Section 5, so you want this operation to feel automatic.

### 3.5 Matrix Addition and Scalar Multiplication

These work exactly like their vector counterparts and require no new ideas: add corresponding entries, or multiply every entry by a scalar.

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$$

The interesting matrix operation is multiplication, which is **not** entrywise. That is the next section.


## 4. Matrix Multiplication

Matrix multiplication is the operation that everyone who has studied basic math most often misremember, so we will take it slowly and build it up: rule first, intuition second, then a fully worked example.

### 4.1 The Dimension Rule

You can only multiply two matrices if their inner dimensions match.

$$\underbrace{A}_{m \times n} \cdot \underbrace{B}_{n \times p} = \underbrace{C}_{m \times p}$$

Read the rule: an $m \times n$ matrix times an $n \times p$ matrix produces an $m \times p$ matrix. The two $n$s (the **inner** dimensions) must match. The two **outer** dimensions ($m$ and $p$) become the dimensions of the result.

```
   Dimensions of A:   m x n
   Dimensions of B:       n x p
                          ^^^
                  these must match
   
   Result dimensions:  m x p
```

If the inner dimensions do not match, the multiplication is simply not defined.

### 4.2 The Computation Rule

Each entry of the product $C = AB$ is the **dot product** of a row of $A$ with a column of $B$:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

In plain English: to compute entry $(i, j)$ of the result, take row $i$ of $A$, take column $j$ of $B$, and dot-product them.

This is exactly where the dot product from Section 2.5 reappears. Matrix multiplication is really just a structured collection of dot products.

### 4.3 Worked Example by Hand

Let

$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \qquad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

Both are $2 \times 2$, so the inner dimensions match (both 2) and the product will be $2 \times 2$.

**Entry $C_{11}$ (top-left):** row 1 of $A$ dotted with column 1 of $B$:

$$C_{11} = (1)(5) + (2)(7) = 5 + 14 = 19$$

**Entry $C_{12}$ (top-right):** row 1 of $A$ dotted with column 2 of $B$:

$$C_{12} = (1)(6) + (2)(8) = 6 + 16 = 22$$

**Entry $C_{21}$ (bottom-left):** row 2 of $A$ dotted with column 1 of $B$:

$$C_{21} = (3)(5) + (4)(7) = 15 + 28 = 43$$

**Entry $C_{22}$ (bottom-right):** row 2 of $A$ dotted with column 2 of $B$:

$$C_{22} = (3)(6) + (4)(8) = 18 + 32 = 50$$

Putting them together:

$$AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

> **Tip:** if you ever get lost, write the dimensions above each matrix. The whole rule collapses into "the two middle numbers match, the two outside numbers stick around." Once that becomes automatic, the rest is mechanical.

### 4.4 Properties (and One Crucial Non-Property)

Matrix multiplication is:

| Property | Rule | Comment |
|---|---|---|
| **Associative** | $(AB)C = A(BC)$ | You can group multiplications however you like |
| **Distributive** | $A(B + C) = AB + AC$ | Standard algebra still works for the distributive side |
| **Identity** | $AI = IA = A$ | The identity matrix acts like the number 1 |

But here is the **crucial non-property**: matrix multiplication is **NOT commutative**. In general, $AB \neq BA$. The order in which you multiply matrices matters, and changing the order will almost always give a different answer (or may not even be defined at all if the dimensions stop matching). This is a common source of bugs, so be deliberate.

### 4.5 Matrix-Vector Multiplication: The Key Special Case

When you multiply a matrix by a vector, treat the vector as a matrix with one column. An $m \times n$ matrix times an $n \times 1$ vector produces an $m \times 1$ vector.

**This is the operation behind linear regression predictions.** If $X$ is your design matrix (one row per sample, one column per feature) and $\vec{w}$ is your weight vector, then $X \vec{w}$ is a vector of predictions, one per sample. Every prediction is just a dot product of a row of $X$ with $\vec{w}$.


## 5. The Normal Equations: Linear Algebra Powering Linear Regression

Now we use everything from sections 2 to 4 to solve the central problem of supervised learning: given a dataset $X$ and target values $\vec{y}$, find the weights $\vec{w}$ that produce the best predictions.

### 5.1 The Linear Regression Setup

We want to model:

$$\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n$$

In matrix form, this becomes the elegant expression:

$$\hat{\vec{y}} = X \vec{w}$$

where:

- $X$ is the **design matrix**: an $m \times (n+1)$ matrix with one row per sample, one column per feature, and a leading column of 1s for the intercept term $w_0$
- $\vec{w}$ is the **weight vector** of length $n+1$
- $\hat{\vec{y}}$ is the vector of predictions of length $m$

The actual target values are $\vec{y}$, and the **residuals** (errors) are $\vec{y} - X\vec{w}$.

### 5.2 The Least Squares Problem

The "best" weights are the ones that make the predictions as close to the true values as possible. Formally, we want to find $\vec{w}$ that minimises the **sum of squared residuals**:

$$\text{Loss}(\vec{w}) = \|\vec{y} - X\vec{w}\|^2 = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

This is exactly the squared L2 norm of the residual vector. Minimising it is called **least squares**.

### 5.3 The Normal Equations

There is a remarkable result: this minimisation problem has a **closed-form solution** that requires no iteration, no learning rate, and no optimiser. It comes from setting the derivative of the loss to zero (a calculus step we will revisit tomorrow). The result is the **normal equations**:

$$X^T X \vec{w} = X^T \vec{y}$$

Solving for $\vec{w}$:

$$\boxed{\vec{w} = (X^T X)^{-1} X^T \vec{y}}$$

That single equation is one of the most consequential in all of applied mathematics. It tells you exactly which weights produce the best-fit line (or hyperplane, in higher dimensions) for your data.

### 5.4 Geometric Intuition

Here is the picture that makes the normal equations make sense. Think of all possible predictions $X\vec{w}$ as you vary $\vec{w}$; they sweep out a flat surface in $m$-dimensional space (the **column space** of $X$). Your actual targets $\vec{y}$ live somewhere in that space, usually not exactly on the surface.

The least squares solution is the point on the surface **closest** to $\vec{y}$. Geometrically, that point is the **perpendicular projection** of $\vec{y}$ onto the column space. The residual vector (the difference between $\vec{y}$ and its projection) must be **orthogonal** to every column of $X$.

Orthogonality of two vectors means their dot product is zero. Saying the residual is orthogonal to every column of $X$ is exactly the same as saying:

$$X^T (\vec{y} - X\vec{w}) = \vec{0}$$

Rearranging gives the normal equations. The whole derivation is really just "the best-fit residual must be perpendicular to the feature directions," dressed up in matrix notation.

### 5.5 When Are the Normal Equations Appropriate?

The closed form requires inverting $X^T X$, which is an $(n+1) \times (n+1)$ matrix. For small to medium feature counts (say, $n < 10{,}000$), this is fast and exact. For very large feature counts or when $X^T X$ is singular (e.g. with collinear features), the inversion becomes impractical or impossible, and we turn to iterative methods like **gradient descent**, the focus of tomorrow's lesson.

## 6. Hands-On: Linear Algebra with the California Housing Dataset

We will work with the **California Housing dataset**, which contains 20,640 housing records from the 1990 California census with 8 numeric features. This is genuinely large compared to Iris (150 rows) and gives the matrix operations real substance.

### 6.1 Setup and Loading

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

sns.set(style="whitegrid")
np.random.seed(42)

# Load the dataset
data = fetch_california_housing()
X_full = pd.DataFrame(data.data, columns=data.feature_names)
y_full = pd.Series(data.target, name="MedHouseVal")

print("Feature names:", list(X_full.columns))
print(f"X shape: {X_full.shape}")
print(f"y shape: {y_full.shape}")
X_full.head()
```

> The 8 features are: `MedInc` (median income), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, and `Longitude`. The target `MedHouseVal` is the median house value in $100,000s.

### 6.2 The Dataset as a Matrix, Rows as Vectors

```python
# A single row of the dataset is a feature vector in 8-dimensional space
sample_vector = X_full.iloc[0].values
print(f"First sample (a vector in R^8): {sample_vector}")
print(f"Its dimension: {len(sample_vector)}")

# The full dataset is a 20640 x 8 matrix
X_matrix = X_full.values
print(f"\nDataset matrix shape: {X_matrix.shape}")
print(f"That is {X_matrix.shape[0]:,} rows (samples) and {X_matrix.shape[1]} columns (features)")
```

### 6.3 Vector Operations on Real Data

**Magnitude of feature vectors.** This tells us how "large" each sample's combined feature vector is.

```python
# Compute L2 norm (magnitude) of each sample's feature vector
sample_norms = np.linalg.norm(X_matrix, axis=1)
print(f"Mean magnitude across all 20,640 samples: {sample_norms.mean():.2f}")
print(f"Smallest magnitude: {sample_norms.min():.2f}")
print(f"Largest magnitude:  {sample_norms.max():.2f}")

plt.figure(figsize=(9, 4))
sns.histplot(sample_norms, bins=50, color="steelblue")
plt.title("Distribution of L2 Norms across 20,640 Samples")
plt.xlabel("||sample vector||")
plt.ylabel("Count")
plt.show()
```

> **Note on scale:** without feature scaling, the `Population` column dominates the magnitude calculation. This is one motivation for standardisation, which we will apply when we run regression below.

**Dot product as similarity between samples.**

```python
# Compare the first sample to all others using the dot product
first_sample = X_matrix[0]
similarities = X_matrix @ first_sample   # Matrix-vector multiplication, 20640 dot products

# Which 5 samples are most similar (largest dot product)?
top_indices = np.argsort(similarities)[-5:][::-1]
print("Top 5 most similar samples to sample 0 (by raw dot product):")
print(X_full.iloc[top_indices])
```

### 6.4 Matrix Operations on Real Data

**Transpose.**

```python
X_T = X_matrix.T
print(f"X shape:   {X_matrix.shape}")
print(f"X^T shape: {X_T.shape}")
print("Transposing flips rows and columns: 20640 x 8 becomes 8 x 20640")
```

**The Gram matrix $X^T X$.** This $8 \times 8$ matrix appears directly in the normal equations and is closely related to the covariance matrix from Monday.

```python
gram = X_T @ X_matrix
print(f"X^T X is {gram.shape[0]} x {gram.shape[1]}")
print(f"Is it symmetric? {np.allclose(gram, gram.T)}")

plt.figure(figsize=(7, 5))
sns.heatmap(gram, annot=False, cmap="coolwarm", center=0)
plt.title("Gram Matrix X^T X (8 x 8)")
plt.show()
```

### 6.5 Implementing Linear Regression with the Normal Equations

Now we apply the boxed equation from Section 5.3 directly to predict California house values.

```python
# Step 1: Standardise features so they share a comparable scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_matrix)

# Step 2: Train/test split for honest evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full.values, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set:     {X_test.shape[0]:,} samples")
```

```python
# Step 3: Add a column of 1s for the intercept term w_0
def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_train_design = add_intercept(X_train)
X_test_design = add_intercept(X_test)

print(f"Design matrix shape (with intercept column): {X_train_design.shape}")
```

```python
# Step 4: Apply the normal equations: w = (X^T X)^(-1) X^T y
XtX = X_train_design.T @ X_train_design
XtX_inv = np.linalg.inv(XtX)
Xty = X_train_design.T @ y_train
w = XtX_inv @ Xty

print("Learned weights via normal equations:")
feature_names = ["Intercept"] + list(X_full.columns)
for name, weight in zip(feature_names, w):
    print(f"  {name:<12s}: {weight:+.4f}")
```

```python
# Step 5: Make predictions and evaluate
y_pred_train = X_train_design @ w
y_pred_test = X_test_design @ w

print(f"Training R^2: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test R^2:     {r2_score(y_test, y_pred_test):.4f}")
print(f"Test RMSE:    {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
```

**Verifying against scikit-learn:** if our linear algebra is correct, sklearn's `LinearRegression` should produce essentially identical coefficients.

```python
sk_model = LinearRegression()
sk_model.fit(X_train, y_train)

print("Comparison of our weights vs. sklearn (after the intercept):")
for name, ours, sk in zip(feature_names[1:], w[1:], sk_model.coef_):
    print(f"  {name:<12s}: ours = {ours:+.4f}   sklearn = {sk:+.4f}")
print(f"\n  Intercept     : ours = {w[0]:+.4f}   sklearn = {sk_model.intercept_:+.4f}")
```

The two should agree to several decimal places. Any small difference is floating-point numerical error, not a real mismatch.

### 6.6 Visualising Predictions

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y_test, y_pred_test, alpha=0.2, color="steelblue")
axes[0].plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             color="black", linestyle="--", label="Perfect prediction")
axes[0].set_xlabel("Actual MedHouseVal")
axes[0].set_ylabel("Predicted MedHouseVal")
axes[0].set_title("Predicted vs Actual (Test Set)")
axes[0].legend()

residuals = y_test - y_pred_test
sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="tomato")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].set_xlabel("Residual (actual - predicted)")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.show()
```

### 6.7 What the Coefficients Tell Us

```python
coef_df = pd.DataFrame({
    "Feature": feature_names[1:],
    "Weight": w[1:]
}).sort_values("Weight", key=abs, ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(data=coef_df, x="Weight", y="Feature",
            palette=["seagreen" if v > 0 else "tomato" for v in coef_df["Weight"]])
plt.axvline(0, color="black", lw=0.8)
plt.title("Standardised Regression Coefficients (California Housing)")
plt.show()
```

Because we standardised the features in Step 1, the coefficients are directly comparable. `MedInc` is overwhelmingly the most important positive predictor of house value, while `Latitude` and `Longitude` carry meaningful negative weights, reflecting that location matters in California in ways the other features cannot capture.


## Daily Challenge

1. **Vector practice by hand.** Given $\vec{a} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ and $\vec{b} = \begin{bmatrix} 4 \\ -1 \\ 2 \end{bmatrix}$, compute by hand: (a) $\vec{a} + \vec{b}$, (b) $3\vec{a}$, (c) $\vec{a} \cdot \vec{b}$, (d) $\|\vec{a}\|$. Then verify each with NumPy.

2. **Matrix multiplication by hand.** Compute $AB$ for
$$A = \begin{bmatrix} 2 & 1 \\ 0 & 3 \\ 1 & 4 \end{bmatrix}, \qquad B = \begin{bmatrix} 5 & 2 \\ 1 & 3 \end{bmatrix}$$
Verify with `A @ B`. What is the shape of the result, and why?

3. **Normal equations on a different dataset.** Repeat the regression workflow from Section 6.5 on the **Diabetes dataset** (`sklearn.datasets.load_diabetes`). Standardise the features, split into train/test, apply the normal equations, and report training and test R². Which feature has the largest absolute weight?

4. **Why standardise?** Repeat the California Housing regression *without* the StandardScaler step. Are the predictions different? Are the coefficients still comparable across features? Explain in 2 to 3 sentences.

5. **Bonus: The Gram matrix and the correlation matrix.** Compute the correlation matrix of the *standardised* California Housing features. Then compute $\frac{1}{n} X^T X$ where $X$ is also the standardised matrix. How are these two matrices related? Why?


## Summary

| Concept | Notation | What it does |
|---|---|---|
| Vector | $\vec{v}$ | Ordered list of $n$ numbers; a point or arrow in $\mathbb{R}^n$ |
| Vector addition | $\vec{a} + \vec{b}$ | Component-by-component sum |
| Scalar multiplication | $c \vec{v}$ | Scale every component by $c$ |
| Dot product | $\vec{a} \cdot \vec{b}$ | $\sum a_i b_i$; measures alignment / similarity |
| Norm (magnitude) | $\|\vec{v}\|$ | Euclidean length: $\sqrt{\sum v_i^2}$ |
| Matrix | $A$ | Rectangular grid of numbers; rows are samples, columns are features |
| Transpose | $A^T$ | Flip rows and columns |
| Matrix multiplication | $AB$ | Entry $(i,j)$ is dot product of row $i$ of $A$ with column $j$ of $B$ |
| Identity matrix | $I$ | Acts like the number 1; $AI = A$ |
| Normal equations | $\vec{w} = (X^T X)^{-1} X^T \vec{y}$ | Closed-form least-squares solution |


## What's Next

We now move into **Calculus Essentials**. You will learn what a derivative is, what a gradient is, and how the same minimisation problem we solved in closed form today can also be solved *iteratively* using gradient descent, which is the algorithm behind essentially every modern machine learning model. The normal equations gave us *one specific answer*; calculus will show us the *general procedure* that scales to problems where no closed form exists.


*Lesson designed by Zindua School Team members, co-authored by Claude and Counterchecked by Zindua School Data Science Team*
