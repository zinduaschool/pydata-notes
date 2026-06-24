# Distributions

> **Course:** Data Science Foundations
> **Lesson:** Day 2, Distributions
> **Prerequisites:** Statistics Fundamentals

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what a distribution is and why it sits at the centre of every statistical analysis you will ever run
- Distinguish between discrete and continuous distributions, and identify which mathematical function (PMF or PDF) describes each
- Identify and describe the Uniform, Poisson, and Normal distributions in terms of their shapes, parameters, and real-world use cases
- Recognise each of these distributions visually from a histogram with reasonable confidence
- Apply the 68-95-99.7 rule to real problems and explain why it works
- Plot and identify distributions in Python using Matplotlib and Seaborn on a substantial real-world dataset

## 1. What is a Distribution and Why Does it Matter?

A **distribution** describes how the values in a dataset are spread out. It answers the question:

> *"Where do most of the values sit, and how often do different values appear?"*

Think back to Monday's lesson. Mean, median, mode, variance, and standard deviation are all *numbers* that summarise a dataset. A distribution is the bigger picture those numbers come from, the full *shape* of the data. Two datasets can share the same mean and the same standard deviation and still have very different distributions, as you saw on Monday when we introduced skewness and kurtosis.

### 1.1 Why Distributions Are the Foundation of Everything

Understanding a distribution lets you do four critical things:

1. **Know what is normal versus what is unusual** in your data. If you know exam scores follow a Normal distribution with mean 70 and standard deviation 10, then a score of 95 is unambiguously rare. Without a distribution, "95 is high" is a vague intuition; with a distribution, you can quantify *how* high.
2. **Choose the right statistical tools.** Many statistical tests (which you will meet on Thursday) assume your data follows a specific distribution, almost always the Normal. Applying those tools to data that does not satisfy the assumption gives wrong answers.
3. **Make probability statements.** Once you know the distribution, you can answer questions like "what is the probability of seeing a value this extreme?" This is the focus of Wednesday's lesson.
4. **Spot anomalies.** Outliers are values that do not fit the expected distribution. Fraud detection, quality control, and intrusion detection all rest on this idea.

### 1.2 A Roadmap for Today

We will start with a foundational distinction (discrete versus continuous), then walk through three of the most important specific distributions in data science. We will use a substantial real dataset for the hands-on section, the Diamonds dataset, which contains over 53,000 records and lets us see distributions on a scale that is closer to what you will encounter in real work.

## 2. Discrete versus Continuous Distributions

Before looking at specific distributions, it is important to understand the two broad families they fall into, because this determines *how* you calculate probabilities from them.

### 2.1 Discrete Distributions

A variable is **discrete** when it can only take specific, separate (countable) values, usually whole numbers. You can list every possible outcome.

**Examples.** Number of customer support tickets in an hour. Number of defective items in a batch. The result of a die roll. The number of children in a household.

Discrete distributions are described by a **Probability Mass Function (PMF)**, which gives the *exact* probability of each individual value:

$$P(X = k)$$

Because outcomes are countable, it makes sense to ask, "what is the probability of exactly 3 calls?" and get a real, non-zero answer.

### 2.2 Continuous Distributions

A variable is **continuous** when it can take *any* value within a range. There are infinitely many possible values, even within a small interval.

**Examples.** Height. Weight. Exam scores (in principle). Time until an event happens. The price of a diamond.

Continuous distributions are described by a **Probability Density Function (PDF)**. Critically, because there are infinitely many possible exact values, the probability of any single exact value is technically zero. You never ask, "what is the probability that a diamond costs exactly $5,247.83?" You ask, "what is the probability a diamond costs between $5,000 and $6,000?" This corresponds to the **area under the PDF curve** between the two bounds:

$$P(a \leq X \leq b) = \int_a^b f(x)\,dx$$

You do not need to solve this integral by hand. `scipy.stats` provides functions (mainly `cdf`) that do this for you. The intuition is what matters: with continuous data, you always ask about *ranges*, not exact values.

### 2.3 The Key Distinction at a Glance

| | Discrete | Continuous |
|---|---|---|
| **Possible values** | Countable (0, 1, 2, 3, ...) | Any value in a range |
| **Function** | PMF, Probability Mass Function | PDF, Probability Density Function |
| **Typical question** | "What is the probability of exactly k?" | "What is the probability between a and b?" |
| **Examples in this lesson** | Poisson | Uniform, Normal |

> **A subtle case worth flagging.** The **Uniform distribution** can technically exist in both discrete form (a fair die: 1 through 6) and continuous form (a random number between 0 and 1). We will focus on the continuous case below, since it is the more common one in data analysis. The intuition (every value equally likely within the range) is the same in both cases.

## 3. Uniform Distribution

### 3.1 What It Is

In a uniform distribution, **every value within a defined range has an equal chance of occurring**. No value is more likely than any other, and the resulting PDF is perfectly flat across the range.

The continuous Uniform distribution on the interval [a, b] has PDF:

$$f(x) = \begin{cases} \frac{1}{b - a}, & a \leq x \leq b \\ 0, & \text{otherwise} \end{cases}$$

The height $1 / (b - a)$ is chosen so that the total area under the curve equals 1, as it must for any valid probability distribution. Its mean is the midpoint $(a + b) / 2$, a property worth remembering.

### 3.2 Real-World Examples

- **Rolling a fair die.** Each face (1 through 6) has exactly a 1-in-6 chance.
- **Random number generators.** When software generates a number between 0 and 1, every value in that range is equally likely.
- **Bus arrival times.** If a bus arrives every 30 minutes and you show up at a random time, your wait time is uniformly distributed between 0 and 30 minutes.
- **Lottery draws.** Each number has an equal chance of being picked.

### 3.3 Shape and How to Identify It

A uniform distribution produces a **flat, rectangular histogram**. Every bar is roughly the same height, with no peak anywhere. This is the easiest distribution shape to spot: if your histogram looks like a flat table-top rather than a hill, you are likely looking at a uniform distribution.

Parameters: defined by a minimum and a maximum, often written as `Uniform(a, b)`. There is no "centre of gravity" in the same sense as a peaked distribution. Every outcome in `[a, b]` is equally weighted.

### 3.4 When to Watch For It

If you see a flat distribution in your data where you expected variation, it is worth investigating *why*. Possibilities include:

- The data was **randomly generated or shuffled** (simulated test data, randomised assignment to groups)
- There genuinely is **no meaningful pattern** to explain, every outcome really is equally likely
- Your **data collection process** is biased toward equal representation in a way that masks a real underlying pattern (quota sampling can do this)

In real data analysis, exactly uniform variables are rare. Approximately uniform variables usually mean someone (or something) artificially balanced the data.

## 4. Poisson Distribution

### 4.1 What It Is

The Poisson distribution models the **number of times an event occurs in a fixed interval of time or space**, when events happen independently of one another and at a known average rate. It answers the question: *"How many times will this happen in this interval?"*

It is defined by a single parameter:

- **λ (lambda)** is the average number of events per interval.

The PMF is:

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

You will not compute this by hand often. `scipy.stats.poisson` does it for you. The reason this formula matters is that it lets you turn "events happen at an average rate of 4 per hour" into precise probabilities like "what is the chance we get exactly 7 events next hour?"

### 4.2 Real-World Examples

- Number of customer support tickets received per hour
- Number of emails arriving in an inbox per day
- Number of accidents at an intersection per month
- Number of website visits per minute
- Number of defects found per batch of products
- Number of insurance claims per policyholder per year

### 4.3 Shape and How to Identify It

- Values are always **non-negative whole numbers**. You cannot have 2.5 accidents, so the PMF only has bars at 0, 1, 2, 3, and so on.
- For **small λ**, the distribution is noticeably **right-skewed**: a tall bar near zero, with a tail trailing off to the right.
- As **λ grows larger**, the shape gradually becomes more symmetric and starts to resemble a Normal distribution. This is a useful visual cue for estimating roughly how large λ is just by looking at the shape.
- A unique mathematical property: the **mean and variance are both equal to λ**. If you compute a dataset's mean and variance and find they are approximately equal, that is a strong hint you may be looking at Poisson-distributed data. This property is called **equidispersion**, and real-world count data often violates it (overdispersion is common), but the Poisson is still the standard starting point.

### 4.4 A Practical Example

If a call centre receives an average of **4 calls per hour (λ = 4)**, the Poisson distribution lets you calculate things like:

- The probability of receiving exactly 6 calls in the next hour
- The probability of receiving 0 calls (a quiet hour)
- Whether a sudden spike to 12 calls is statistically unusual or just normal variation

This is extremely useful for **staffing, capacity planning, and anomaly detection**. You will come back to exactly this kind of question when you compute Poisson probabilities directly in Wednesday's lesson.

## 5. Normal (Gaussian) Distribution

### 5.1 What It Is

The Normal distribution is the most important distribution in statistics, and arguably in all of applied mathematics. It is the classic **bell curve**: most values cluster around the mean, and values become progressively less frequent the further you move away from it in either direction.

It is defined by exactly two parameters:

- **Mean (μ)** is where the centre of the bell sits.
- **Standard deviation (σ)** controls how wide or narrow the bell is.

The PDF is:

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

The formula looks intimidating, but the shape it produces is the familiar bell. The key insight is that two numbers (μ and σ) completely determine the distribution. You will rarely write this PDF out yourself, since `scipy.stats.norm` handles it for you.

### 5.2 Real-World Examples

- Exam scores in a large class
- Heights and weights within a population
- Measurement errors in scientific experiments
- Blood pressure readings
- IQ scores (by construction)

### 5.3 Shape and How to Identify It

A Normal distribution is **symmetric** around its mean, **bell-shaped**, with a single peak at the centre (unimodal), and tails that taper off smoothly and never quite reach zero. Visually, if you fold the histogram in half at the centre, the two sides should roughly mirror each other. If one side has a noticeably longer tail than the other, the data is skewed rather than Normal (recall skewness from Monday's lesson).

> **A caution.** A histogram that *looks* roughly bell-shaped is not proof of Normality on its own. Later in the course you will meet formal tools like the QQ plot and the Shapiro-Wilk test. For now, visual inspection combined with the empirical rule below is enough for most purposes.

### 5.4 The 68-95-99.7 Rule (Empirical Rule)

This rule describes how data is spread around the mean in a Normal distribution:

| Range | Percentage of data |
|---|---|
| Within **1 standard deviation** (μ ± 1σ) | ~**68%** of values |
| Within **2 standard deviations** (μ ± 2σ) | ~**95%** of values |
| Within **3 standard deviations** (μ ± 3σ) | ~**99.7%** of values |

This rule is so useful that it is worth memorising. It lets you make rough probability statements in your head without reaching for a calculator. For example, if exam scores are Normal with mean 70 and standard deviation 10, you can immediately say:

- 68% of students scored between **60 and 80**
- 95% of students scored between **50 and 90**
- 99.7% of students scored between **40 and 100**

A student scoring **95** is more than 2 standard deviations above the mean, placing them in roughly the top 2.5% of the class. That is a precise, quantifiable statement derived directly from the shape of the distribution.

### 5.5 Why the Normal Distribution Matters So Much

There are two reasons the Normal distribution dominates statistics. First, many statistical tools (including linear regression and the t-tests on Thursday) **assume the underlying data, or the model residuals, is approximately Normal**. Knowing whether your data fits this shape is a critical check before applying those tools. Second, the **Central Limit Theorem** (previewed Monday) shows that sample means tend toward a Normal distribution regardless of the original population's shape. This is precisely why the Normal distribution shows up so often even when the raw data itself is not Normal. The averages we work with in inferential statistics are well-behaved even when the underlying data is messy.

## 6. Comparing the Three Distributions

| Distribution | Type | Shape | Key Parameter(s) | Common Use Case |
|---|---|---|---|---|
| **Uniform** | Continuous (or discrete) | Flat, rectangular | min, max | Random processes, equal-probability outcomes |
| **Poisson** | Discrete | Right-skewed for small λ; more symmetric for large λ | λ (rate) | Event counts, arrivals, defects |
| **Normal** | Continuous | Symmetric bell curve | μ (mean), σ (std) | Natural phenomena, test scores, measurement error |

A quick visual gut-check for identifying which distribution you are looking at:

1. **Is the histogram flat with no peak?** Likely Uniform.
2. **Are the values whole numbers with a clear peak and a tail trailing to the right?** Likely Poisson.
3. **Is the histogram symmetric with a single peak in the middle, tapering off smoothly on both sides?** Likely Normal.

If none of these fit, your data may follow one of the many other distributions you will meet later (Exponential, Beta, Log-Normal, Gamma, and so on), or it may be a mixture of multiple distributions, which is exactly what causes the multimodal shapes you saw with Penguins on Monday.

## 7. Hands-On: Python with the Diamonds Dataset

For today's hands-on work we will use the **Diamonds dataset** from seaborn, which contains 53,940 diamond records with 10 features including price, carat, cut, color, clarity, and physical dimensions. This is genuinely heavy compared to most teaching datasets, and it gives us authentic distributions to look at rather than only simulated ones.

### 7.1 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
np.random.seed(42)

diamonds = sns.load_dataset("diamonds")
print(f"Shape: {diamonds.shape}")
print(f"\nColumns: {list(diamonds.columns)}")
print(f"\nFirst rows:\n{diamonds.head()}")
```

```python
# Quick descriptive snapshot
print(diamonds.describe().round(2))
```

The dataset has roughly 54,000 diamonds, the price ranges from $326 to $18,823, and carats range from 0.2 to 5.0. These numbers alone hint that the price distribution is likely right-skewed (most diamonds are inexpensive, with a long tail of expensive ones).

### 7.2 Theoretical (Simulated) Distributions

Before we look at the real data, let us build intuition by simulating each of the three distributions and plotting them.

#### Uniform

```python
uniform_data = np.random.uniform(low=0, high=100, size=5000)

plt.figure(figsize=(8, 4))
sns.histplot(uniform_data, bins=30, kde=True, color="steelblue")
plt.title("Simulated Uniform Distribution, n = 5,000")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print(f"Mean: {uniform_data.mean():.2f}")
print(f"Variance: {uniform_data.var():.2f}")
```

What to look for: a flat histogram, with the KDE line nearly horizontal across the range.

#### Poisson

```python
lam = 4
poisson_data = np.random.poisson(lam=lam, size=5000)

plt.figure(figsize=(8, 4))
sns.histplot(poisson_data, bins=15, kde=False, color="darkorchid", discrete=True)
plt.axvline(lam, color='red', linestyle='--', label=f'λ = {lam}')
plt.title("Simulated Poisson Distribution, λ = 4, n = 5,000")
plt.xlabel("Number of Events")
plt.ylabel("Frequency")
plt.legend()
plt.show()

print(f"Mean: {poisson_data.mean():.2f}")
print(f"Variance: {poisson_data.var():.2f}")
print("Notice that mean ≈ variance, which is the Poisson signature.")
```

Try re-running with `lam = 30` and observe how much more symmetric (Normal-looking) the shape becomes.

#### Normal

```python
normal_data = np.random.normal(loc=70, scale=10, size=5000)

mu, sigma = normal_data.mean(), normal_data.std()

plt.figure(figsize=(8, 4))
sns.histplot(normal_data, bins=40, kde=True, color="seagreen")
plt.axvline(mu, color='red', linestyle='--', label='Mean')
plt.axvline(mu + sigma, color='orange', linestyle='--', label='+1 SD')
plt.axvline(mu - sigma, color='orange', linestyle='--', label='-1 SD')
plt.title("Simulated Normal Distribution, μ = 70, σ = 10, n = 5,000")
plt.legend()
plt.show()

# Verify the empirical rule
within_1sd = np.mean((normal_data >= mu - sigma) & (normal_data <= mu + sigma))
within_2sd = np.mean((normal_data >= mu - 2*sigma) & (normal_data <= mu + 2*sigma))
within_3sd = np.mean((normal_data >= mu - 3*sigma) & (normal_data <= mu + 3*sigma))

print(f"Within 1 SD: {within_1sd:.1%} (expected ~68%)")
print(f"Within 2 SD: {within_2sd:.1%} (expected ~95%)")
print(f"Within 3 SD: {within_3sd:.1%} (expected ~99.7%)")
```

The percentages should land very close to the theoretical values. Increasing `size` makes the convergence tighter.

### 7.3 Side-by-Side Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(uniform_data, bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Uniform")

sns.histplot(poisson_data, bins=15, discrete=True, ax=axes[1], color="darkorchid")
axes[1].set_title("Poisson (λ=4)")

sns.histplot(normal_data, bins=40, kde=True, ax=axes[2], color="seagreen")
axes[2].set_title("Normal (μ=70, σ=10)")

plt.suptitle("The Three Shapes Side by Side", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

### 7.4 Real Distributions in the Diamonds Dataset

Now let us look at the actual distributions of several Diamonds features. The reason this is more educational than only looking at simulations: real data is messier, and recognising shapes in real data is the skill that matters.

```python
fig, axes = plt.subplots(2, 2, figsize=(13, 8))

# Carat: a famously right-skewed distribution
sns.histplot(diamonds['carat'], bins=80, kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title(f"Carat (skewness = {diamonds['carat'].skew():.2f})")
axes[0, 0].set_xlim(0, 3)

# Price: also strongly right-skewed
sns.histplot(diamonds['price'], bins=80, kde=True, ax=axes[0, 1], color="tomato")
axes[0, 1].set_title(f"Price, $ (skewness = {diamonds['price'].skew():.2f})")

# Depth: roughly Normal
sns.histplot(diamonds['depth'], bins=60, kde=True, ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title(f"Depth, % (skewness = {diamonds['depth'].skew():.2f})")
axes[1, 0].set_xlim(55, 70)

# Table: roughly Normal but with discreteness artefacts
sns.histplot(diamonds['table'], bins=60, kde=True, ax=axes[1, 1], color="darkorchid")
axes[1, 1].set_title(f"Table, % (skewness = {diamonds['table'].skew():.2f})")
axes[1, 1].set_xlim(50, 70)

plt.tight_layout()
plt.show()
```

Look at what the data is telling you. `carat` and `price` are massively right-skewed: many cheap small diamonds, few expensive large ones. `depth` is roughly bell-shaped. `table` is bell-ish but with visible "stripes" because measurements cluster at whole percentage values.

```python
# Numeric summary of skewness across all numeric features
numeric_cols = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']
summary = pd.DataFrame({
    'mean': diamonds[numeric_cols].mean(),
    'median': diamonds[numeric_cols].median(),
    'skewness': diamonds[numeric_cols].skew(),
    'kurtosis': diamonds[numeric_cols].kurt(),
})
print(summary.round(3))
```

### 7.5 A Classic Trick: The Log Transformation

When data is right-skewed and positive (which is common for prices, populations, incomes, and counts), taking the **logarithm** often produces a distribution that looks much more Normal. This is so common in practice that it is worth showing here.

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sns.histplot(diamonds['price'], bins=80, kde=True, ax=axes[0], color="tomato")
axes[0].set_title(f"Price (skewness = {diamonds['price'].skew():.2f})")

log_price = np.log(diamonds['price'])
sns.histplot(log_price, bins=80, kde=True, ax=axes[1], color="seagreen")
axes[1].set_title(f"log(Price) (skewness = {log_price.skew():.2f})")

plt.tight_layout()
plt.show()
```

The transformed distribution is approximately Normal, which means it can now be modelled with tools that assume Normality (like ordinary least squares regression). This is one of the most common preprocessing tricks in practice.

### 7.6 Real Counts: Searching for a Poisson Shape

Diamonds itself is mostly continuous data, so to see a real Poisson-like shape we will manufacture one from the data by counting how many diamonds fall into specific price bins.

```python
# Count diamonds in each $100 price bin
price_bins = np.arange(diamonds['price'].min(), diamonds['price'].max(), 100)
diamonds['price_bin'] = pd.cut(diamonds['price'], bins=price_bins)
counts_per_bin = diamonds.groupby('price_bin', observed=True).size().values

# This is not strictly Poisson, but it has the right flavour: counts of events per bin
plt.figure(figsize=(9, 4))
sns.histplot(counts_per_bin, bins=40, color="darkorchid")
plt.title("Distribution of Diamond Counts per $100 Price Bin")
plt.xlabel("Number of Diamonds in Bin")
plt.ylabel("Number of Bins")
plt.show()

print(f"Mean of counts:     {counts_per_bin.mean():.2f}")
print(f"Variance of counts: {counts_per_bin.var():.2f}")
```

A pure Poisson would have mean ≈ variance. Real-world count data often has variance much larger than the mean (called overdispersion), and you can see hints of that here. This is one of the most common practical reasons people reach for a Negative Binomial distribution instead of a Poisson, which you may meet in a future course.

### 7.7 Cut Quality, A Real Discrete Variable

```python
plt.figure(figsize=(9, 4))
sns.countplot(data=diamonds, x='cut', order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
plt.title("Distribution of Cut Quality across 53,940 Diamonds")
plt.show()

print(diamonds['cut'].value_counts())
```

This is a discrete categorical distribution, not strictly any of the three we covered, but it shows how to think about distributions for categorical data: you visualise the counts (or proportions) per category.

## Daily Challenge

1. Generate your own dataset of 5,000 samples each using `np.random.uniform()`, `np.random.poisson()`, and `np.random.normal()` with parameters of your choice. Plot all three on a single figure with subplots, clearly labelled. Report the mean and variance of each.
2. For your Poisson sample, confirm that mean and variance are approximately equal. Then regenerate the sample with a much larger λ (try 50) and describe how the shape changes. At what value of λ does the Poisson distribution start to look essentially Normal to your eye?
3. For your Normal sample, verify the 68-95-99.7 rule and report the actual percentages you observe.
4. Using the Diamonds dataset, pick three numeric features and determine for each: is it approximately Normal, right-skewed, left-skewed, or none of these? Use both a histogram and the numeric skewness value to justify your answer.
5. Apply the log transformation to one right-skewed feature in the Diamonds dataset. Report the skewness before and after. Did the transformation help?
6. **Bonus.** A factory logs the number of defects per batch over the past 200 batches. Without running any code, what distribution would you expect this data to follow, and why? Then generate a simulated dataset matching that expectation and verify visually.

> Submit your notebook to the Kanban board under your name by end of day.

## Summary

| Concept | Question it answers | Key idea |
|---|---|---|
| **Distribution** | How are values in my data spread out? | The full shape behind summary statistics |
| **Discrete distribution** | What is the probability of exactly this count? | Described by a PMF |
| **Continuous distribution** | What is the probability of falling in this range? | Described by a PDF, area under the curve |
| **Uniform** | Is every outcome equally likely? | Flat histogram, min and max parameters |
| **Poisson** | How many events occur in a fixed interval? | Right-skewed for small λ, mean = variance = λ |
| **Normal** | Where do most values cluster, symmetrically? | Bell curve, defined by μ and σ, 68-95-99.7 rule |
| **Log transformation** | How do I un-skew positive right-skewed data? | Apply `np.log` and recheck the shape |

## What's Next

Tomorrow's lesson moves into **Probability Theory**, where we will formalise exactly how to calculate the probabilities these distributions describe. We will cover joint and conditional probability, and Bayes' theorem for updating beliefs as new evidence arrives. You will see the Uniform, Poisson, and Normal distributions again, this time as tools for answering precise probability questions.

*Lesson designed for internal use, Statistical Foundations, Data Science Curriculum*
