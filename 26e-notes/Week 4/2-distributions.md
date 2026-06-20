# Distributions

> **Course:** Data Analytics
> **Lesson:** Data Distributions
> **Prerequisites:** Statistical Foundations (mean, std, variance)


---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what a distribution is and why it matters in data analysis
- Distinguish between discrete and continuous distributions, and the functions used to describe each
- Identify and describe the Uniform, Poisson, and Normal distributions — their shapes, parameters, and real-world use cases
- Recognise these distributions visually from a histogram
- Apply the 68-95-99.7 rule to real problems
- Plot and identify distributions in Python using Matplotlib and Seaborn

---

## 1. What is a Distribution and Why Does it Matter?

A **distribution** describes how the values in a dataset are spread out. It answers the question:

> *"Where do most of the values sit, and how often do different values appear?"*

Think back to Monday's lesson: mean, median, mode, variance, and standard deviation are all *numbers* that summarise a dataset. A distribution is the bigger picture those numbers come from — the full *shape* of the data. Two datasets can share the same mean and standard deviation and still have very different distributions, as you saw with skewness on Monday.

### Why it matters

Understanding a distribution lets you:

- Know what's **normal** vs. what's **unusual** in your data
- Choose the **right statistical tools** — many statistical tests (which you'll meet on Thursday) assume your data follows a specific distribution
- Make **probability statements** — e.g. "what's the probability of seeing a value this extreme?" (the focus of Wednesday's lesson)
- Spot **anomalies** — values that don't fit the expected pattern

Distributions are the bridge between Monday's descriptive statistics and the probability and inference tools coming the rest of this week.

---

## 2. Discrete vs. Continuous Distributions

Before looking at specific distributions, it's important to understand the two broad families they fall into — because this determines *how* you calculate probabilities from them.

### 2.1 Discrete Distributions

A variable is **discrete** when it can only take specific, separate (countable) values — usually whole numbers. You can list every possible outcome.

**Examples:** number of customer support tickets in an hour, number of defective items in a batch, the result of a die roll.

Discrete distributions are described by a **Probability Mass Function (PMF)**, which gives the *exact* probability of each individual value:

$$P(X = k)$$

Because outcomes are countable, it makes sense to ask "what is the probability of exactly 3 calls?" — and get a real, non-zero answer.

### 2.2 Continuous Distributions

A variable is **continuous** when it can take *any* value within a range — there are infinitely many possible values, even within a small interval.

**Examples:** height, weight, exam scores (in principle), time until an event happens.

Continuous distributions are described by a **Probability Density Function (PDF)**. Critically, because there are infinitely many possible exact values, the probability of any *single exact value* is technically zero. Instead, you compute the probability of a **range** of values, which corresponds to the *area under the curve* between two points:

$$P(a \leq X \leq b) = \int_a^b f(x)\,dx$$

You don't need to solve this integral by hand — `scipy.stats` will do it for you — but the intuition matters: with continuous data, you always ask "what's the probability of falling *between* these two values?" rather than "what's the probability of exactly this value?"

### 2.3 The Key Distinction at a Glance

| | Discrete | Continuous |
|---|---|---|
| **Possible values** | Countable (e.g. 0, 1, 2, 3...) | Any value in a range |
| **Function** | PMF — Probability Mass Function | PDF — Probability Density Function |
| **Typical question** | "What's the probability of exactly k?" | "What's the probability between a and b?" |
| **Examples in this lesson** | Poisson | Uniform, Normal |

>  Note that the **Uniform distribution** can technically exist in both discrete form (e.g. a fair die: 1–6) and continuous form (e.g. a random number between 0 and 1). We'll focus on the continuous case below, since it's the more common one in data analysis.

---

## 3. Uniform Distribution

### 3.1 What It Is

In a uniform distribution, **every value within a defined range has an equal chance of occurring**. No value is more likely than any other — everything is flat and even.

### 3.2 Real-World Examples

- **Rolling a fair die** — each face (1–6) has exactly a 1-in-6 chance
- **Random number generators** — when software generates a number between 0 and 1, every value in that range is equally likely
- **Bus arrival times** — if a bus arrives every 30 minutes and you show up at a random time, your wait time is uniformly distributed between 0 and 30 minutes
- **Lottery draws** — each number has an equal chance of being picked

### 3.3 Shape and How to Identify It

A uniform distribution produces a **flat, rectangular histogram** — every bar is roughly the same height, with no peak anywhere. This is the easiest distribution shape to spot: if your histogram looks like a flat table-top rather than a hill, you're likely looking at a uniform distribution.

**Parameters:** defined by a minimum and maximum value, often written as `Uniform(a, b)`. There is no "centre of gravity" — every outcome in `[a, b]` is equally weighted.

### 3.4 When to Watch For It

If you see a flat distribution in your data where you expected variation, it's worth investigating *why*. It could mean:
- The data was **randomly generated or shuffled** (e.g. simulated test data, randomised assignment)
- There genuinely is **no meaningful pattern** to explain — every outcome really is equally likely
- Your **data collection process** is biased toward equal representation in a way that masks a real underlying pattern (e.g. quota sampling)

---

## 4. Poisson Distribution

### 4.1 What It Is

The Poisson distribution models the **number of times an event occurs in a fixed interval of time or space**, when events happen independently of one another and at a known average rate. It answers the question: *"How many times will this happen?"*

It is defined by a single parameter:
- **λ (lambda)** — the average number of events per interval

### 4.2 Real-World Examples

- Number of **customer support tickets** received per hour
- Number of **emails** arriving in an inbox per day
- Number of **accidents** at an intersection per month
- Number of **website visits** per minute
- Number of **defects** found per batch of products

### 4.3 Shape and How to Identify It

- Values are always **non-negative whole numbers** — you can't have 2.5 accidents, so the PMF only has bars at 0, 1, 2, 3...
- For **small λ**, the distribution is noticeably **right-skewed**: a tall bar near zero, with a tail trailing off to the right
- As **λ gets larger**, the shape gradually becomes more symmetric and starts to resemble a normal distribution — this is a useful visual cue for estimating roughly how large λ is just by looking at the shape
- A unique mathematical property: the **mean and variance are both equal to λ**. If you compute a dataset's mean and variance and find they're approximately equal, that's a strong hint you may be looking at Poisson-distributed data

### 4.4 Practical Example

If a call centre receives an average of **4 calls per hour (λ = 4)**, the Poisson distribution lets you calculate things like:
- The probability of receiving exactly 6 calls in the next hour
- The probability of receiving 0 calls (a quiet hour)
- Whether a sudden spike to 12 calls is statistically unusual or just normal variation

This is extremely useful for **staffing, capacity planning, and anomaly detection** — and you'll come back to exactly this kind of question when you compute Poisson probabilities directly in Wednesday's lesson.

---

## 5. Normal (Gaussian) Distribution

### 5.1 What It Is

The normal distribution is the most important distribution in statistics. It's the classic **bell curve** — most values cluster around the mean, and values become progressively less frequent the further you move away from it in either direction.

It is defined by exactly two parameters:
- **Mean (μ)** — where the centre of the bell sits
- **Standard deviation (σ)** — how wide or narrow the bell is

### 5.2 Real-World Examples

- Exam scores in a large class
- Heights and weights within a population
- Measurement errors in scientific experiments
- Blood pressure readings

### 5.3 Shape and How to Identify It

A normal distribution is **symmetric** around its mean, **bell-shaped**, with a single peak at the centre (unimodal), and tails that taper off smoothly and never quite reach zero. Visually: if you fold the histogram in half at the centre, the two sides should roughly mirror each other. If one side has a noticeably longer tail than the other, the data is skewed rather than normal (recall skewness from Monday's lesson).

>  A histogram that *looks* roughly bell-shaped isn't proof of normality on its own — later in the course you'll meet formal tools like the QQ plot and statistical normality tests. For now, visual inspection combined with the empirical rule below is enough.

### 5.4 The 68-95-99.7 Rule (Empirical Rule)

This rule describes how data is spread around the mean in a normal distribution:

| Range | Percentage of data |
|---|---|
| Within **1 standard deviation** (μ ± 1σ) | ~**68%** of values |
| Within **2 standard deviations** (μ ± 2σ) | ~**95%** of values |
| Within **3 standard deviations** (μ ± 3σ) | ~**99.7%** of values |

### 5.5 Practical Example

Say exam scores are normally distributed with a mean of **70** and a standard deviation of **10**:

- 68% of students scored between **60 and 80**
- 95% of students scored between **50 and 90**
- 99.7% of students scored between **40 and 100**

A student scoring **95** is more than 2 standard deviations above the mean — placing them in roughly the top 2.5% of the class. That's a precise, quantifiable statement derived directly from the shape of the distribution.

### 5.6 Why It Matters So Much

Many statistical tools — including linear regression and the t-tests you'll encounter on Thursday — **assume the underlying data (or model residuals) is approximately normally distributed**. Knowing whether your data fits this shape, and roughly why, is a critical check before applying those tools. It's also the distribution that the Central Limit Theorem (introduced briefly on Monday) shows sample means tend toward, regardless of the shape of the original population — which is precisely why the normal distribution shows up so often even when the raw data itself isn't normal.

---

## 6. Comparing the Three Distributions

| Distribution | Type | Shape | Key Parameter(s) | Common Use Case |
|---|---|---|---|---|
| **Uniform** | Continuous (or discrete) | Flat, rectangular | min, max | Random processes, equal-probability outcomes |
| **Poisson** | Discrete | Right-skewed for small λ; more symmetric for large λ | λ (rate) | Event counts, arrivals, defects |
| **Normal** | Continuous | Symmetric bell curve | μ (mean), σ (std) | Natural phenomena, test scores, measurement error |

A quick visual gut-check for identifying which distribution you're looking at:

1. **Is the histogram flat with no peak?** → Likely Uniform
2. **Are the values whole numbers with a clear peak and a tail trailing to the right?** → Likely Poisson
3. **Is the histogram symmetric with a single peak in the middle, tapering off smoothly on both sides?** → Likely Normal

---

## 7. Hands-On: Python

### 7.1 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
np.random.seed(42)
```

---

### 7.2 Uniform Distribution

```python
uniform_data = np.random.uniform(low=0, high=100, size=1000)

plt.figure(figsize=(8, 4))
sns.histplot(uniform_data, bins=20, kde=True, color="steelblue")
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print(f"Mean: {uniform_data.mean():.2f}")
print(f"Variance: {uniform_data.var():.2f}")
```

> **What to look for:** A flat histogram — all bars roughly equal height. The KDE line should be nearly horizontal across the range.

---

### 7.3 Poisson Distribution

```python
# Discrete distribution: average of 4 events per interval
lam = 4
poisson_data = np.random.poisson(lam=lam, size=1000)

plt.figure(figsize=(8, 4))
sns.histplot(poisson_data, bins=15, kde=False, color="darkorchid", discrete=True)
plt.axvline(lam, color='red', linestyle='--', label=f'λ = {lam}')
plt.title("Poisson Distribution (Calls per Hour, λ=4)")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Mean: {poisson_data.mean():.2f}")
print(f"Variance: {poisson_data.var():.2f}")
print("Notice: mean and variance are approximately equal — a Poisson signature.")
```

> **What to look for:** A right-skewed, discrete distribution. Most counts cluster near λ=4, with a long tail trailing to the right. Try re-running with `lam=30` and observe how much more symmetric (normal-looking) the shape becomes.

---

### 7.4 Normal Distribution

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

# Verify the empirical rule
mu, sigma = normal_data.mean(), normal_data.std()
within_1sd = np.mean((normal_data >= mu - sigma) & (normal_data <= mu + sigma))
within_2sd = np.mean((normal_data >= mu - 2*sigma) & (normal_data <= mu + 2*sigma))
within_3sd = np.mean((normal_data >= mu - 3*sigma) & (normal_data <= mu + 3*sigma))

print(f"Within 1 SD: {within_1sd:.1%} (expected ~68%)")
print(f"Within 2 SD: {within_2sd:.1%} (expected ~95%)")
print(f"Within 3 SD: {within_3sd:.1%} (expected ~99.7%)")
```

> **What to look for:** A symmetric bell shape centred on the mean. The red and orange lines visualise the 68-95-99.7 rule, and the printed percentages should land close to the theoretical values.

---

### 7.5 Side-by-Side Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(uniform_data, bins=20, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Uniform")

sns.histplot(poisson_data, bins=15, discrete=True, ax=axes[1], color="darkorchid")
axes[1].set_title("Poisson (λ=4)")

sns.histplot(normal_data, bins=30, kde=True, ax=axes[2], color="seagreen")
axes[2].set_title("Normal (μ=70, σ=10)")

plt.suptitle("Comparing Distribution Shapes", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

---

### 7.6 Identifying Distributions on a Real Dataset

```python
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

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
> - Do any features show two separate humps? *(Hint: the dataset contains 3 species — mixed groups can distort a single distribution's shape, a pattern you'll want to keep in mind any time a "normal-looking" variable looks oddly lumpy.)*
> - Are any features skewed, and does that match the skewness values you'd compute using Monday's `df.skew()`?

---

## Daily Challenge

1. Generate your own dataset using `np.random.uniform()`, `np.random.poisson()`, and `np.random.normal()` with parameters of your choice. Plot all three on a single figure with subplots, clearly labelled.
2. For your Poisson sample, compute the mean and variance and confirm they're approximately equal. Then regenerate the sample with a much larger λ (e.g. 50) and describe how the shape changes.
3. For your Normal sample, verify the 68-95-99.7 rule the way Section 7.4 does, and report the actual percentages you observe.
4. Using the Iris dataset, pick **one feature** and determine: is it Uniform, Poisson-like, or Normal (or none of these)? Write 3–4 sentences justifying your answer using both the shape of the histogram and at least one summary statistic from Monday's lesson (mean, variance, or skewness).
5. **Bonus:** A factory logs the number of defects per batch over the past 200 batches. Without running any code, what distribution would you expect this data to follow, and why? Then generate a simulated dataset to check your reasoning.

> Submit your notebook to the Kanban board under your name by end of day.

---

## Summary

| Concept | Question it answers | Key idea |
|---|---|---|
| **Distribution** | How are values in my data spread out? | The full shape behind summary statistics |
| **Discrete distribution** | What's the probability of exactly this count? | Described by a PMF |
| **Continuous distribution** | What's the probability of falling in this range? | Described by a PDF; area under the curve |
| **Uniform** | Is every outcome equally likely? | Flat histogram; min/max parameters |
| **Poisson** | How many events occur in a fixed interval? | Right-skewed for small λ; mean = variance = λ |
| **Normal** | Where do most values cluster, symmetrically? | Bell curve; defined by μ and σ; 68-95-99.7 rule |

---

## What's Next

Tomorrow's lesson moves into **Probability Theory** — formalising exactly how to calculate the probabilities these distributions describe, including joint and conditional probability, and Bayes' theorem for updating beliefs as new evidence arrives. You'll see the Uniform, Poisson, and Normal distributions again, this time as tools for answering precise probability questions.

*Lesson designed by Zindua School Team members, co-authored by Claude and Counterchecked by Zindua School Data Science Team*