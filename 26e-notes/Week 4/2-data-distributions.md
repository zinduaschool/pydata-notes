# Data Distributions

> **Course:** Data Science Foundations
> **Lesson:** Data Distributions
> **Prerequisites:** Statistical Foundations (mean, std, variance)

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what a distribution is and why it matters in data analysis
- Identify and describe four key distribution types: Uniform, Normal, Poisson, and Bimodal
- Recognise the 68-95-99.7 rule and apply it to real problems
- Plot and interpret distributions in Python using Matplotlib and Seaborn

---

## 1. What is a Distribution and Why Does it Matter?

A **distribution** describes how values in a dataset are spread out. It answers the question:

> *"Where do most of the values sit, and how often do different values appear?"*

Think of it this way — if you collected the heights of 1,000 people and plotted them, you'd notice a pattern. Most people cluster around an average height, with fewer people on the very short or very tall ends. That pattern *is* the distribution.

### Why it matters

Understanding a distribution lets you:

- Know what's **normal** vs what's **unusual** in your data
- Choose the **right statistical tools** (many assume a specific distribution)
- Make **predictions** — e.g. "what's the probability of X happening?"
- Spot **anomalies** — values that don't fit the expected pattern

Distributions are the foundation of everything from fraud detection to quality control to machine learning model assumptions.

---

## 2. Uniform Distribution

### What it is

In a uniform distribution, **every value has an equal chance of occurring**. No value is more likely than any other — everything is flat and even.

### Real-world examples

- **Rolling a fair die** — each face (1–6) has exactly a 1 in 6 chance
- **Random number generators** — when software generates a random number between 0 and 1, every value is equally likely
- **Bus arrival times** — if a bus arrives every 30 minutes and you show up at a random time, your wait time is uniformly distributed between 0 and 30 minutes
- **Lottery draws** — each number has an equal chance of being picked

### What it looks like

A uniform distribution produces a **flat, rectangular histogram** — all bars are roughly the same height.

### When to watch for it

If you see a flat distribution in your data where you expected variation, it could mean:
- The data was **randomly generated or shuffled**
- There is **no meaningful pattern** to explain
- Your **data collection was biased** toward equal representation

---

## 3. Gaussian (Normal) Distribution

### What it is

The normal distribution is the most important distribution in statistics. It's the classic **bell curve** — most values cluster around the mean, and values become less frequent as you move further away in either direction.

It is defined by just two parameters:
- **Mean (μ)** — where the center of the bell sits
- **Standard Deviation (σ)** — how wide or narrow the bell is

### Real-world examples

- Exam scores in a large class
- Heights and weights of a population
- Measurement errors in experiments
- Blood pressure readings

### The 68-95-99.7 Rule (Empirical Rule)

This rule tells you how data is spread around the mean in a normal distribution:

| Range | Percentage of data |
|---|---|
| Within **1 standard deviation** (μ ± 1σ) | ~**68%** of values |
| Within **2 standard deviations** (μ ± 2σ) | ~**95%** of values |
| Within **3 standard deviations** (μ ± 3σ) | ~**99.7%** of values |

### Practical example

Say exam scores are normally distributed with a mean of **70** and a standard deviation of **10**:

- 68% of students scored between **60 and 80**
- 95% of students scored between **50 and 90**
- 99.7% of students scored between **40 and 100**

A student scoring **95** is more than 2 standard deviations above the mean — in the top ~2.5% of the class. That's a meaningful, quantifiable statement.

### Why it matters so much

Many machine learning algorithms and statistical tests (e.g. linear regression, t-tests) **assume your data is normally distributed**. Knowing whether your data fits this shape is critical before applying those tools.

---

## 4. Poisson Distribution

### What it is

The Poisson distribution models the **number of times an event occurs in a fixed interval of time or space**, when events happen independently and at a known average rate.

It answers questions like: *"How many times will this happen?"*

It is defined by one parameter:
- **λ (lambda)** — the average number of events per interval

### Real-world examples

- Number of **customer support tickets** received per hour
- Number of **emails** arriving in your inbox per day
- Number of **accidents** at an intersection per month
- Number of **website visits** per minute
- Number of **defects** found per batch of products

### Key properties

- Values are always **whole numbers** (counts — you can't have 2.5 accidents)
- The distribution is **right-skewed** for small λ, but approaches normal as λ gets large
- The **mean and variance are both equal to λ** — a unique property

### Practical example

If a call centre receives an average of **4 calls per hour (λ = 4)**, you can use the Poisson distribution to calculate:
- The probability of receiving exactly 6 calls in the next hour
- The probability of receiving 0 calls (a quiet hour)
- Whether a sudden spike to 12 calls is unusual or expected

This is extremely useful for **staffing, capacity planning, and anomaly detection**.

---

## 5. Bimodal Distribution

### What it is

A bimodal distribution has **two distinct peaks** (modes) rather than one. It's a signal that your data contains **two different groups or patterns** mixed together.

"Bi" means two — two humps, two clusters, two stories hiding in the same dataset.

### Real-world examples

- **Exam scores** where half the class studied and half didn't — two clusters form
- **Heights** in a dataset that mixes men and women without labelling them
- **Product ratings** that are either very high or very low (love it or hate it)
- **Traffic speeds** on a road — one cluster for normal traffic, one for rush hour

### What it signals

A bimodal distribution is often a sign that you should **split your data into segments** and analyse each group separately. Treating a bimodal dataset as a single normal distribution gives you a misleading average that doesn't represent either group well.

> **Example:** If men average 175cm and women average 162cm, the combined average of ~168cm accurately describes nobody. Segmenting reveals the real story.

### How to spot it

Look for a histogram with **two clear humps** separated by a visible dip. It's one of the most visually obvious distribution shapes.

---

## 6. Hands-On: Plotting and Identifying Distributions in Python

### Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
np.random.seed(42)
```

---

### 6.1 Uniform Distribution

```python
uniform_data = np.random.uniform(low=0, high=100, size=1000)

plt.figure(figsize=(8, 4))
sns.histplot(uniform_data, bins=20, kde=True, color="steelblue")
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```

> **What to look for:** A flat histogram — all bars roughly equal height. The KDE line will be nearly horizontal.

---

### 6.2 Normal Distribution

```python
normal_data = np.random.normal(loc=70, scale=10, size=1000)  # mean=70, std=10

plt.figure(figsize=(8, 4))
sns.histplot(normal_data, bins=30, kde=True, color="seagreen")
plt.axvline(np.mean(normal_data), color='red', linestyle='--', label='Mean')
plt.axvline(np.mean(normal_data) + np.std(normal_data), color='orange', linestyle='--', label='+1 SD')
plt.axvline(np.mean(normal_data) - np.std(normal_data), color='orange', linestyle='--', label='-1 SD')
plt.title("Normal Distribution (Exam Scores)")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()
```

> **What to look for:** A symmetric bell shape centered on the mean. The red and orange lines help you visualise the 68-95-99.7 rule.

---

### 6.3 Poisson Distribution

```python
poisson_data = np.random.poisson(lam=4, size=1000)  # avg 4 events per interval

plt.figure(figsize=(8, 4))
sns.histplot(poisson_data, bins=15, kde=False, color="darkorchid", discrete=True)
plt.title("Poisson Distribution (Calls per Hour, λ=4)")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.show()
```

> **What to look for:** A right-skewed, discrete distribution. Most counts are near λ=4, with a long tail to the right.

---

### 6.4 Bimodal Distribution

```python
group_a = np.random.normal(loc=55, scale=8, size=500)   # students who didn't study
group_b = np.random.normal(loc=80, scale=6, size=500)   # students who studied
bimodal_data = np.concatenate([group_a, group_b])

plt.figure(figsize=(8, 4))
sns.histplot(bimodal_data, bins=40, kde=True, color="tomato")
plt.title("Bimodal Distribution (Exam Scores — Two Groups)")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()
```

> **What to look for:** Two distinct humps with a visible dip between them — a clear signal of two underlying groups.

---

### 6.5 Identifying Distributions on a Real Dataset

```python
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Plot all four features
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    sns.histplot(df[col], bins=25, kde=True, ax=axes[i], color="teal")
    axes[i].set_title(f"Distribution of {col}")

plt.suptitle("Iris Dataset — Feature Distributions", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

> **Interpret what you see:**
> - Which features look approximately normal?
> - Which appear bimodal? What might explain that? *(Hint: there are 3 species in this dataset)*
> - Are any features skewed?

---

## Daily Challenge

1. Generate your own dataset using `np.random.normal()`, `np.random.uniform()`, and `np.random.poisson()` with parameters of your choice
2. Plot all three on a single figure with subplots — label each clearly
3. Using the Iris dataset, pick **one feature** and determine: is it normal, skewed, or bimodal? Write 3–4 sentences justifying your answer
4. **Bonus:** Create a bimodal distribution by combining two groups from the Iris dataset (e.g. one species vs another) and plot it

> Submit your notebook to the Kanban board under your name by end of day.

---

## Summary

| Distribution | Shape | Key parameter(s) | Common use case |
|---|---|---|---|
| **Uniform** | Flat | min, max | Random processes, equal probability |
| **Normal** | Bell curve | μ (mean), σ (std) | Natural phenomena, test scores, errors |
| **Poisson** | Right-skewed, discrete | λ (rate) | Event counts, arrivals, defects |
| **Bimodal** | Two humps | — | Mixed populations, segmented data |

---

## What's Next

In the next lesson we'll move into **correlation and causation** — building on covariance to explore how variables relate, and the critical difference between a statistical relationship and a real-world cause.

---

*Lesson designed for internal use — Statistical Foundations, Data Science Curriculum*