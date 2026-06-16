# Statistics Fundamentals

> **Course:** Data Science Foundations
> **Lesson:** Day 1 — Statistics Fundamentals
> **Prerequisites:** Basic Python, NumPy/Pandas familiarity

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Compute and interpret the mean, median, and mode, and choose the right measure for a given dataset
- Quantify how spread out a dataset is using range, variance, standard deviation, and the interquartile range
- Measure and interpret relationships between variables using covariance and correlation
- Combine these measures into a coherent **descriptive statistics** summary of a dataset, including its shape
- Understand what **inferential statistics** is, how it differs from description, and why it matters
- Apply all of the above in Python on a real dataset

---

## 1. Measures of Central Tendency

Central tendency answers the question: **"What is the typical value in this dataset?"** There are three common measures, and each tells a slightly different story.

### 1.1 Mean (Average)

The mean is the sum of all values divided by the number of values.

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

**Example:**
Scores: `[55, 60, 70, 80, 85]`
Mean = (55 + 60 + 70 + 80 + 85) / 5 = **70**

**When to use it:** When your data is roughly symmetric with no extreme outliers. The mean uses every value in the dataset, which makes it informative — but also sensitive. A single extreme value can pull it far from where most of the data actually sits.

### 1.2 Median

The median is the **middle value** once the data is sorted. If there's an even number of values, it's the average of the two middle values.

**Example:**
Scores: `[55, 60, 70, 80, 85]` → Median = **70**
Scores: `[55, 60, 70, 80]` → Median = (60 + 70) / 2 = **65**

**When to use it:** When your data has outliers or is skewed. Household income is the classic example — a handful of very high earners can drag the mean far above what a "typical" household earns, but the median stays close to the middle of the pack.

### 1.3 Mode

The mode is the **most frequently occurring value**. A dataset can have one mode, several modes, or none at all.

**Example:**
Scores: `[55, 60, 60, 70, 80, 80, 80, 85]` → Mode = **80**

**When to use it:** Mostly for categorical data — the most common product category purchased, the most frequent browser type, the most common survey response.

### Choosing Between Mean, Median, and Mode

| Situation | Best Measure |
|---|---|
| Symmetric data, no outliers | Mean |
| Skewed data or outliers present | Median |
| Categorical data, or finding the most common value | Mode |

> 💡 A useful habit: compute **all three**. If the mean and median are close together, your data is roughly symmetric. If they diverge significantly, that's your first clue that the data is skewed — and you'll formalise exactly what "skewed" means in Section 4.

---

## 2. Measures of Spread

Central tendency tells you where the data is centred. But two datasets can share the same mean and look completely different — one tightly clustered, one widely scattered. **Spread** tells you how much the values vary around that centre.

### 2.1 Range

The simplest measure of spread — the distance between the largest and smallest value.

$$\text{Range} = x_{max} - x_{min}$$

**Limitation:** The range depends entirely on two values (the extremes), so a single outlier can make it misleading.

### 2.2 Variance

Variance measures the **average squared distance** of each value from the mean.

$$\sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}$$

**Why squared?** If you didn't square the differences, the positive and negative deviations from the mean would always cancel out to zero — variance would be meaningless.

**Example:**
Scores: `[60, 70, 80]`, Mean = 70
Variance = [(60−70)² + (70−70)² + (80−70)²] / 3
= [100 + 0 + 100] / 3 = **66.7**

**The catch:** Variance is in *squared* units (e.g. "squared marks"), which is hard to interpret intuitively. That's where standard deviation comes in.

### 2.3 Standard Deviation

Standard deviation is simply the **square root of the variance**, bringing the measure back into the original units.

$$\sigma = \sqrt{\sigma^2}$$

From the example above: Standard Deviation = √66.7 ≈ **8.16**

This means most scores sit roughly **8 marks away from the mean** — far more interpretable than "66.7 squared marks."

> 💡 **Population vs. Sample:** When your data represents the *entire* population, divide by `n` (this is what the formulas above show). When your data is a *sample* drawn from a larger population — which is the far more common case in practice — divide by `n − 1` instead. This is called **Bessel's correction**, and it corrects for the fact that a sample tends to slightly underestimate the true population variance. Pandas uses `n − 1` by default; NumPy uses `n` by default.

### 2.4 Interquartile Range (IQR) and the Five-Number Summary

Variance and standard deviation describe spread around the *mean* — but if your data is skewed, the mean itself may not be a great reference point. The **five-number summary** gives a more robust picture:

| Statistic | Meaning |
|---|---|
| Minimum | Smallest value |
| Q1 (25th percentile) | 25% of data falls below this |
| Median (Q2, 50th percentile) | The middle value |
| Q3 (75th percentile) | 75% of data falls below this |
| Maximum | Largest value |

The **Interquartile Range (IQR)** is the spread of the middle 50% of the data:

$$\text{IQR} = Q3 - Q1$$

The IQR is the basis for one of the most common outlier detection rules — **Tukey's fences**: any value below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR` is flagged as a potential outlier. This is exactly what a box plot visualises.

### 2.5 Coefficient of Variation — Comparing Spread Across Scales

Standard deviation is in the same units as your data, which makes it hard to compare spread across variables measured on different scales (e.g. "is income or age more variable?"). The **coefficient of variation (CV)** normalises standard deviation by the mean:

$$CV = \frac{\sigma}{\bar{x}}$$

Often expressed as a percentage. A CV of 10% means the standard deviation is 10% of the mean — useful for comparing relative variability between datasets with very different units or magnitudes.

---

## 3. Relationships Between Variables

So far, every measure has looked at **one variable at a time**. Covariance and correlation ask a different question: **"Do two variables move together?"**

### 3.1 Covariance

$$\text{Cov}(X, Y) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{n}$$

| Covariance Value | Meaning |
|---|---|
| **Positive** | When X increases, Y tends to increase |
| **Negative** | When X increases, Y tends to decrease |
| **Near zero** | X and Y show little linear relationship |

**Example — Hours Studied (X) vs. Exam Score (Y):**

| Student | Hours Studied (X) | Score (Y) |
|---|---|---|
| A | 2 | 50 |
| B | 4 | 65 |
| C | 6 | 80 |
| D | 8 | 90 |

As hours increase, scores increase → **positive covariance**.

### 3.2 Correlation (Pearson's r)

The problem with covariance is that its value depends on the **scale** of the variables — covariance between two variables measured in thousands will be much larger than the same relationship measured in single digits, even if the underlying relationship is identical. This makes covariance values hard to compare across datasets.

**Correlation** solves this by normalising covariance by the standard deviations of both variables, producing a value that always falls between **−1 and +1**:

$$r = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

| Correlation (r) | Interpretation |
|---|---|
| `r = +1` | Perfect positive linear relationship |
| `r = 0` | No linear relationship |
| `r = -1` | Perfect negative linear relationship |
| `0.7 ≤ \|r\| < 1` | Strong relationship |
| `0.3 ≤ \|r\| < 0.7` | Moderate relationship |
| `\|r\| < 0.3` | Weak relationship |

Because it's unitless and bounded, correlation is what you'll use almost everywhere in practice — covariance is mostly a stepping stone toward it (and reappears later as the building block of the covariance *matrix* used in linear algebra).

### 3.3 Correlation ≠ Causation

A strong correlation tells you two variables move together — it does **not** tell you why. Classic examples: ice cream sales and drowning incidents are strongly correlated (both rise in summer), but ice cream doesn't cause drowning. A third variable — temperature — drives both. This is called a **confounding variable**, and it's one of the most important caveats in all of data analysis. Always ask: *"Could something else be driving both of these?"* before concluding one variable affects the other.

---

## 4. Descriptive Statistics — Putting It All Together

### 4.1 What "Descriptive Statistics" Means

Everything covered in Sections 1–3 — central tendency, spread, and relationships — falls under a single umbrella: **descriptive statistics**. Its job is to *summarise and describe* the data you actually have, with no attempt to generalise beyond it.

If someone asks "how did the class do on the exam?", descriptive statistics gives you the answer: *"The average score was 70%, scores typically ranged within about 8 points of that, and students who studied more tended to score higher (r = 0.92)."* That sentence alone — mean, spread, and relationship — *is* descriptive statistics in action.

### 4.2 Shape: Skewness and Kurtosis

Two datasets can have the same mean, the same standard deviation, and still look completely different in shape. Two additional measures capture this:

**Skewness** measures asymmetry:

| Skewness | Shape | Meaning |
|---|---|---|
| `≈ 0` | Symmetric | Mean ≈ Median |
| `> 0` (right-skewed) | Long tail to the right | Mean > Median (a few very high values pull the mean up) |
| `< 0` (left-skewed) | Long tail to the left | Mean < Median (a few very low values pull the mean down) |

This is exactly why income data is right-skewed — most people earn within a moderate range, but a small number of very high earners stretch the right tail and pull the mean above the median.

**Kurtosis** measures "tailedness" — how much of the data sits in the extreme tails versus the centre, relative to a normal distribution. High kurtosis means more frequent extreme values (heavier tails); low kurtosis means a flatter distribution with fewer extreme values.

### 4.3 Visualising Descriptive Statistics

Numbers alone only tell part of the story — visualisations make patterns immediately obvious:

- **Histogram**: shows the overall shape of the distribution (symmetric, skewed, multimodal)
- **Box plot**: visualises the five-number summary directly — the box spans Q1 to Q3, the line inside is the median, the whiskers extend to the non-outlier extremes, and points beyond the whiskers are flagged outliers (per Tukey's fences from Section 2.4)

You'll build both in the hands-on section below.

---

## 5. Inferential Statistics — A First Look

### 5.1 Population vs. Sample

A **population** is the entire group you're interested in — every student in a country, every customer who has ever used your app. A **sample** is a subset of that population that you actually collect data on, because measuring the entire population is usually impossible or impractical.

### 5.2 Parameters vs. Statistics

This distinction matters more than it might first appear:

| Term | Refers to | Example |
|---|---|---|
| **Parameter** | A numerical summary of the **population** (usually unknown) | The true average score of *every* student who could ever take this exam |
| **Statistic** | A numerical summary of the **sample** (what you actually calculate) | The average score of the 200 students whose data you have |

Everything in Sections 1–4 — when computed on your actual dataset — produces *statistics*. **Inferential statistics** is the set of tools that lets you use a statistic to make a justified statement about the unknown parameter.

### 5.3 Why We Need Inference

Descriptive statistics can tell you that *this sample's* average score was 70. But the questions that actually matter are usually about the bigger picture: *"If we rolled this teaching method out to every student, would the average score really improve?"* You can never test "every student" — so you need a principled way to reason from a sample to a broader conclusion, while being honest about the uncertainty involved.

### 5.4 Preview: What's Coming This Week

Inferential statistics rests on two ideas you'll build over the next few days:

- **Distributions** (Tuesday) — the mathematical models that describe how values are likely to occur, which is what lets us make probability statements about samples in the first place
- **Probability** (Wednesday) — the formal language for quantifying uncertainty and updating beliefs as new evidence arrives
- **Hypothesis testing** (Thursday) — the decision-making framework that combines distributions and probability to answer questions like *"is this difference real, or could it just be chance?"*

A quick preview of the core idea: if you repeatedly drew different samples from the same population and computed the mean of each sample, those sample means would themselves form a distribution — called the **sampling distribution of the mean**. This distribution tends toward a normal shape even when the underlying population isn't normal (a result called the **Central Limit Theorem**), and it's the foundation on which confidence intervals and hypothesis tests are built. You'll see this in action in the hands-on section below — and revisit it formally later in the week.

---

## 6. Hands-On: Python

### 6.1 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
np.random.seed(42)

# We'll use the classic Iris dataset throughout
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
```

---

### 6.2 Measures of Central Tendency

```python
# Mean, median, and mode for all columns
print("Mean:\n", df.mean())
print("\nMedian:\n", df.median())
print("\nMode:\n", df.mode().iloc[0])

# Single column
sepal_length = df['sepal length (cm)']
print("\nSepal length — Mean:", sepal_length.mean())
print("Sepal length — Median:", sepal_length.median())
print("Sepal length — Mode:", sepal_length.mode()[0])
```

---

### 6.3 Measures of Spread

```python
# Range
data_range = sepal_length.max() - sepal_length.min()
print(f"Range: {data_range:.2f}")

# Variance and standard deviation (sample, ddof=1 — Pandas default)
print("\nVariance (sample):\n", df.var())
print("\nStandard Deviation (sample):\n", df.std())

# Population vs sample std
print("\nStd (Population, NumPy default ddof=0):", np.std(sepal_length))
print("Std (Sample, ddof=1):", sepal_length.std())

# Five-number summary
print("\nFive-number summary:\n", sepal_length.describe()[['min', '25%', '50%', '75%', 'max']])

# IQR
q1, q3 = sepal_length.quantile([0.25, 0.75])
iqr = q3 - q1
print(f"\nIQR: {iqr:.2f}")

# Tukey's fences for outlier detection
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = sepal_length[(sepal_length < lower_fence) | (sepal_length > upper_fence)]
print(f"Outlier bounds: [{lower_fence:.2f}, {upper_fence:.2f}]")
print(f"Number of outliers: {len(outliers)}")

# Coefficient of variation
cv = sepal_length.std() / sepal_length.mean()
print(f"\nCoefficient of Variation: {cv:.2%}")
```

---

### 6.4 Relationships: Covariance and Correlation

```python
# Covariance matrix
print("Covariance matrix:\n", df.cov())

# Correlation matrix
print("\nCorrelation matrix:\n", df.corr())

# A specific pair
cov_value = df['sepal length (cm)'].cov(df['petal length (cm)'])
corr_value = df['sepal length (cm)'].corr(df['petal length (cm)'])
print(f"\nCovariance (sepal length, petal length): {cov_value:.4f}")
print(f"Correlation (sepal length, petal length): {corr_value:.4f}")

# Visualise the correlation matrix
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
plt.title("Correlation Matrix — Iris Features")
plt.show()

# Scatter plot of the strongest relationship
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)')
plt.title(f"Sepal Length vs Petal Length (r = {corr_value:.2f})")
plt.show()
```

---

### 6.5 Descriptive Statistics: Shape and Visualisation

```python
# Skewness and kurtosis
print("Skewness:\n", df.skew())
print("\nKurtosis:\n", df.kurt())

# Histogram + box plot side by side for one feature
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df['petal length (cm)'], bins=25, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Histogram — Petal Length")

sns.boxplot(y=df['petal length (cm)'], ax=axes[1], color="seagreen")
axes[1].set_title("Box Plot — Petal Length")

plt.tight_layout()
plt.show()
```

> **Interpret what you see:** Does the histogram look symmetric, skewed, or does it show more than one cluster? Does the box plot flag any outliers? What does the skewness value confirm about the shape?

---

### 6.6 A Taste of Inference: Simulating the Sampling Distribution

```python
# Treat the full Iris dataset as our "population"
population = df['petal length (cm)']
true_mean = population.mean()
print(f"Population mean (the parameter we'd normally never know): {true_mean:.3f}")

# Repeatedly draw small samples and record their means
sample_size = 10
n_simulations = 1000
sample_means = []

for _ in range(n_simulations):
    sample = population.sample(n=sample_size, replace=True)
    sample_means.append(sample.mean())

sample_means = np.array(sample_means)

# Compare the spread of individual values vs. the spread of sample means
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(population, bins=25, kde=True, ax=axes[0], color="steelblue")
axes[0].axvline(true_mean, color='red', linestyle='--', label='Population mean')
axes[0].set_title("Distribution of Individual Values")
axes[0].legend()

sns.histplot(sample_means, bins=25, kde=True, ax=axes[1], color="darkorchid")
axes[1].axvline(true_mean, color='red', linestyle='--', label='Population mean')
axes[1].set_title(f"Sampling Distribution of the Mean (n={sample_size})")
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Standard deviation of individual values: {population.std():.3f}")
print(f"Standard deviation of sample means:      {sample_means.std():.3f}")
```

> 💡 Notice two things: the sampling distribution of the mean is *narrower* than the original distribution (sample means are less variable than individual values), and it looks more bell-shaped — even though the original petal length data may not be perfectly normal. This is the Central Limit Theorem at work, and it's the reason confidence intervals and hypothesis tests (later this week) are even possible.

---

## Daily Challenge

Using the Iris dataset (or a dataset of your choice):

1. Compute the **mean, median, and mode** for at least two columns. Are mean and median close together, or do they diverge? What does that suggest about skewness?
2. Calculate the **standard deviation and IQR** for each column. Which column has the most spread? Are there any outliers by Tukey's fences?
3. Compute the **correlation matrix** and identify the strongest and weakest relationships between features. For the strongest pair, plot a scatter plot.
4. Compute **skewness** for each column and match each value to what you see in a histogram.
5. Run the **sampling distribution simulation** from Section 6.6 using a different sample size (try `n=5` and `n=50`). How does the spread of the sampling distribution change as sample size increases?
6. Write 3–4 sentences summarising your findings as if explaining them to a non-technical colleague — make sure to distinguish what your sample shows from what you could (and couldn't yet) claim about the broader population.

> Submit your notebook to the Kanban board under your name by end of day.

---

## Summary

| Concept | What it measures | Key function |
|---|---|---|
| Mean | Central value (average) | `df.mean()` |
| Median | Middle value | `df.median()` |
| Mode | Most frequent value | `df.mode()` |
| Range | Spread between max and min | `max() - min()` |
| Variance | Average squared spread | `df.var()` |
| Standard Deviation | Typical distance from the mean | `df.std()` |
| IQR | Spread of the middle 50% | `q3 - q1` |
| Coefficient of Variation | Relative spread (unitless) | `std() / mean()` |
| Covariance | Direction of a relationship between two variables | `df.cov()` |
| Correlation | Strength and direction, normalised to [-1, 1] | `df.corr()` |
| Skewness | Asymmetry of the distribution | `df.skew()` |
| Kurtosis | Tailedness of the distribution | `df.kurt()` |
| Population vs. Sample | What you want to know vs. what you can measure | — |
| Sampling Distribution | The distribution of a statistic across repeated samples | Simulation |

---

## What's Next

Tomorrow's lesson moves into **distributions** — the mathematical models that describe how data values are spread, including the Uniform, Poisson, and Normal distributions. These distributions are what make the sampling distribution from Section 6.6 behave the way it does, and they form the foundation for everything in Wednesday's probability lesson and Thursday's hypothesis testing.

---

