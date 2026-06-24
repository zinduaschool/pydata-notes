# Statistics Fundamentals

> **Course:** Data Science Foundations
> **Lesson:** Day 1, Statistics Fundamentals
> **Prerequisites:** Basic Python, NumPy/Pandas familiarity

## Learning Objectives

By the end of this lesson, you will be able to:

- Compute and interpret the mean, median, and mode, and choose the right measure for a given dataset
- Quantify how spread out a dataset is using range, variance, standard deviation, and the interquartile range
- Measure and interpret relationships between variables using covariance and correlation, and explain the difference between them
- Combine these measures into a coherent descriptive statistics summary that includes the shape of the data (skewness and kurtosis)
- Understand what inferential statistics is, how it differs from description, and why every data science project eventually needs it
- Apply all of the above in Python on a real dataset

## 1. Why Statistics Matters in Data Analysis

Data on its own is just noise. Statistics is the toolkit that turns that noise into meaning, and it does so by answering three deceptively simple questions about every dataset you encounter:

1. **What is typical?** Where does the bulk of the data sit?
2. **How much does it vary?** How spread out are the values around what is typical?
3. **What relates to what?** Do certain variables move together, and how strongly?

Without statistics, you are guessing. With statistics, you are reasoning. Every model you will ever build, from a simple linear regression to a deep neural network, ultimately rests on these three questions.

### A grounding example

Imagine you are a data analyst at a school with exam scores for 200 students. Someone asks, *"How did the class do?"* You cannot reasonably read out 200 numbers. But you can say things like:

- *"The average score was 68%."* That is a measure of central tendency.
- *"Most scores were within 10 points of the average."* That is a measure of spread.
- *"Students who attended more sessions scored higher."* That is a relationship.

Those three sentences are statistics in action. Today we make each one precise.

### A note on what comes next

The descriptive tools in Sections 2 to 4 summarise the data you have in hand. The inferential tools previewed in Section 5 use that summary to make justified statements about a wider population you cannot fully observe. Distinguishing the two is one of the most important habits a data scientist develops, so we will be careful to point out which is which throughout the week.

## 2. Measures of Central Tendency

Central tendency answers the question: **"What is the typical value in this dataset?"** There are three common measures, and each tells a slightly different story.

### 2.1 Mean (Average)

The mean is the sum of all values divided by the number of values:

$$\bar{x} = \frac{\sum_{i=1}^{n} x_i}{n}$$

**Worked example.** Scores: `[55, 60, 70, 80, 85]`. Mean = (55 + 60 + 70 + 80 + 85) / 5 = **70**.

**When to use it.** When your data is roughly symmetric with no extreme outliers. The mean uses every value, which makes it informative, but the same property makes it sensitive: a single extreme value can pull it far from where most of the data actually sits.

### 2.2 Median

The median is the **middle value** once the data is sorted. If there is an even number of values, it is the average of the two middle values.

**Worked examples.**

- `[55, 60, 70, 80, 85]` has median **70**
- `[55, 60, 70, 80]` has median (60 + 70) / 2 = **65**

**When to use it.** When your data has outliers or is skewed. Household income is the classic case: a handful of very high earners can drag the mean far above what a "typical" household earns, but the median stays close to the middle of the pack.

### 2.3 Mode

The mode is the **most frequently occurring value**. A dataset can have one mode, several modes, or none at all.

**Worked example.** Scores: `[55, 60, 60, 70, 80, 80, 80, 85]`. Mode = **80**.

**When to use it.** Mostly for categorical data: the most common product category purchased, the most frequent browser type, the most common survey response. For continuous numeric data the mode is rarely interesting because exact repeats are uncommon.

### 2.4 Choosing Between Mean, Median, and Mode

| Situation | Best Measure | Why |
|---|---|---|
| Symmetric data, no outliers | Mean | Uses all the information efficiently |
| Skewed data or outliers present | Median | Robust to extreme values |
| Categorical data, or finding the most common value | Mode | The only one that works for non-numeric categories |

> **A useful habit.** Always compute all three when you can. If the mean and median are close, your data is roughly symmetric. If they diverge significantly, that is your first clue that the data is skewed. We will make "skewed" precise in Section 4.

## 3. Measures of Spread

Central tendency tells you where the data is centred. But two datasets can share the same mean and look completely different (one tightly clustered, one widely scattered). **Spread** tells you how much the values vary around that centre.

### 3.1 Range

The simplest measure of spread is the distance between the largest and smallest value:

$$\text{Range} = x_{\max} - x_{\min}$$

The range depends entirely on the two extremes, so a single outlier can make it misleading. It is useful as a quick orientation but rarely the right answer on its own.

### 3.2 Variance

Variance measures the **average squared distance** of each value from the mean:

$$\sigma^2 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n}$$

**Why squared?** If you did not square the differences, the positive and negative deviations from the mean would always cancel out to zero. Squaring solves that and additionally penalises large deviations more heavily than small ones, which matches how we intuitively think about "spread."

**Worked example.** Scores: `[60, 70, 80]`, mean = 70. Variance = [(60 − 70)² + (70 − 70)² + (80 − 70)²] / 3 = [100 + 0 + 100] / 3 = **66.7**.

**The catch.** Variance is in *squared* units (squared marks, squared dollars, squared kilograms), which is hard to interpret intuitively. That is where standard deviation comes in.

### 3.3 Standard Deviation

Standard deviation is simply the **square root of the variance**, which brings the measure back into the original units:

$$\sigma = \sqrt{\sigma^2}$$

From the example above, standard deviation = √66.7 ≈ **8.16**. This means most scores sit roughly **8 marks away from the mean**, which is far more interpretable than "66.7 squared marks." Standard deviation is by far the most common measure of spread you will encounter in practice.

> **Population versus sample.** When your data represents the *entire* population, divide by `n` (the formulas above show this). When your data is a *sample* drawn from a larger population (the far more common case in practice), divide by `n − 1` instead. This is called **Bessel's correction**, and it compensates for the fact that a sample tends to slightly underestimate the true population variance. Pandas uses `n − 1` by default; NumPy uses `n` by default. This single difference is responsible for many puzzling discrepancies between libraries.

### 3.4 Interquartile Range (IQR) and the Five-Number Summary

Variance and standard deviation describe spread around the *mean*. But if the data is skewed, the mean itself is not a great reference point. The **five-number summary** offers a more robust picture that does not depend on the mean at all:

| Statistic | Meaning |
|---|---|
| Minimum | Smallest value |
| Q1 (25th percentile) | 25% of data falls below this |
| Median (Q2, 50th percentile) | The middle value |
| Q3 (75th percentile) | 75% of data falls below this |
| Maximum | Largest value |

The **Interquartile Range (IQR)** is the spread of the middle 50% of the data:

$$\text{IQR} = Q_3 - Q_1$$

The IQR is also the basis for one of the most common outlier detection rules, known as **Tukey's fences**: any value below `Q1 − 1.5 × IQR` or above `Q3 + 1.5 × IQR` is flagged as a potential outlier. This is exactly what a box plot visualises, and we will use it in the hands-on section below.

### 3.5 Coefficient of Variation: Comparing Spread Across Scales

Standard deviation is in the same units as your data, which makes it hard to compare spread across variables on different scales (is income or age more variable?). The **coefficient of variation (CV)** normalises standard deviation by the mean:

$$CV = \frac{\sigma}{\bar{x}}$$

Often expressed as a percentage. A CV of 10% means the standard deviation is 10% of the mean. This is useful for comparing relative variability between datasets or columns with very different units or magnitudes.

## 4. Relationships Between Variables

So far, every measure has looked at **one variable at a time**. Covariance and correlation ask a different question: **"Do two variables move together?"**

### 4.1 Covariance

$$\text{Cov}(X, Y) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{n}$$

The intuition: for each data point, you look at how far X is from its mean and how far Y is from its mean, multiply those deviations together, then average. If X and Y tend to be both above their means together (and both below together), the products are mostly positive, and covariance is positive. If they move in opposite directions, the products are mostly negative.

| Covariance Value | Meaning |
|---|---|
| **Positive** | When X increases, Y tends to increase |
| **Negative** | When X increases, Y tends to decrease |
| **Near zero** | X and Y show little linear relationship |

**Worked example, Hours Studied (X) versus Exam Score (Y).**

| Student | Hours Studied (X) | Score (Y) |
|---|---|---|
| A | 2 | 50 |
| B | 4 | 65 |
| C | 6 | 80 |
| D | 8 | 90 |

As hours increase, scores increase, so the covariance is positive.

### 4.2 Correlation (Pearson's r)

The problem with covariance is that its value depends on the **scale** of the variables. Covariance between two variables measured in thousands will be much larger than the same relationship measured in single digits, even if the underlying relationship is identical. This makes covariance values hard to compare across datasets.

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

Because it is unitless and bounded, correlation is what you will use almost everywhere in practice. Covariance is mostly a stepping stone toward it, and it reappears later as a building block of the covariance *matrix* used in linear algebra.

### 4.3 An Important Caveat: Correlation Captures Only Linear Relationships

Pearson's correlation specifically measures **linear** association. Two variables can be perfectly related in a non-linear way (for example, $y = x^2$ over a symmetric range) and still have a Pearson correlation near zero. Always plot the data as well as computing a correlation coefficient. The plot will catch what the number alone hides.

### 4.4 Correlation Does Not Mean Causation

A strong correlation tells you two variables move together. It does **not** tell you why. Classic examples include ice cream sales and drowning incidents being strongly correlated (both rise in summer), but ice cream does not cause drowning. A third variable, temperature, drives both. This is called a **confounding variable**, and it is one of the most important caveats in all of data analysis. Always ask, *"Could something else be driving both of these?"* before concluding one variable affects the other.

## 5. Descriptive Statistics: Putting It All Together

### 5.1 What "Descriptive Statistics" Means

Everything covered in Sections 2 to 4 falls under a single umbrella: **descriptive statistics**. Its job is to summarise and describe the data you actually have, with no attempt to generalise beyond it. If someone asks "how did the class do on the exam?", descriptive statistics gives you the answer in a sentence or two.

### 5.2 Shape: Skewness and Kurtosis

Two datasets can have the same mean, the same standard deviation, and still look completely different in shape. Two additional measures capture this difference.

**Skewness** measures asymmetry:

| Skewness | Shape | Meaning |
|---|---|---|
| `≈ 0` | Symmetric | Mean ≈ Median |
| `> 0` (right-skewed) | Long tail to the right | Mean > Median; a few very high values pull the mean up |
| `< 0` (left-skewed) | Long tail to the left | Mean < Median; a few very low values pull the mean down |

This is exactly why income data is right-skewed: most people earn within a moderate range, but a small number of very high earners stretch the right tail and pull the mean above the median.

**Kurtosis** measures "tailedness," meaning how much of the data sits in the extreme tails versus the centre, relative to a normal distribution. High kurtosis means more frequent extreme values (heavier tails). Low kurtosis means a flatter distribution with fewer extreme values. Kurtosis matters because tail behaviour is what drives risk in finance, anomaly detection, and quality control. A distribution that looks "almost normal" but has heavy tails can produce extreme events far more often than a true normal distribution would.

### 5.3 Visualising Descriptive Statistics

Numbers alone tell only part of the story. Visualisations make patterns immediately obvious.

- **Histogram** shows the overall shape of the distribution (symmetric, skewed, multimodal)
- **Box plot** visualises the five-number summary directly. The box spans Q1 to Q3, the line inside is the median, the whiskers extend to the non-outlier extremes, and points beyond the whiskers are flagged outliers per Tukey's fences from Section 3.4
- **Scatter plot** is the right tool for relationships, and pairs naturally with correlation
- **Heatmap of a correlation matrix** is the right tool when you have many variables and want to see all pairwise relationships at once

You will build all of these in the hands-on section.

## 6. Inferential Statistics: A First Look

### 6.1 Population versus Sample

A **population** is the entire group you are interested in, such as every student in a country or every customer who has ever used your app. A **sample** is a subset of that population that you actually collect data on, because measuring the entire population is usually impossible or impractical.

### 6.2 Parameters versus Statistics

This distinction matters more than it might first appear:

| Term | Refers to | Example |
|---|---|---|
| **Parameter** | A numerical summary of the **population** (usually unknown) | The true average score of every student who could ever take this exam |
| **Statistic** | A numerical summary of the **sample** (what you actually calculate) | The average score of the 200 students whose data you have |

Everything in Sections 2 to 5, when computed on your actual dataset, produces *statistics*. **Inferential statistics** is the set of tools that lets you use a statistic to make a justified statement about the unknown parameter.

### 6.3 Why We Need Inference

Descriptive statistics can tell you that this sample's average score was 70. But the questions that actually matter are usually about the bigger picture, such as: *"If we rolled this teaching method out to every student, would the average score really improve?"* You can never test every student, so you need a principled way to reason from a sample to a broader conclusion, while being honest about the uncertainty involved.

### 6.4 Preview: What Is Coming This Week

Inferential statistics rests on two ideas you will build over the next few days:

- **Distributions** (Tuesday) provide the mathematical models that describe how values are likely to occur, which is what lets us make probability statements about samples in the first place.
- **Probability** (Wednesday) gives the formal language for quantifying uncertainty and updating beliefs as new evidence arrives.
- **Hypothesis testing** (Thursday) is the decision-making framework that combines distributions and probability to answer questions like, *"Is this difference real, or could it just be chance?"*

A quick preview of the core idea: if you repeatedly drew different samples from the same population and computed the mean of each sample, those sample means would themselves form a distribution, called the **sampling distribution of the mean**. This distribution tends toward a normal shape even when the underlying population is not normal, a result called the **Central Limit Theorem**. This single fact is the foundation on which confidence intervals and hypothesis tests are built. You will see this in action in the hands-on section below, and revisit it formally later in the week.

## 7. Hands-On: Python with the Penguins Dataset

For today's hands-on work we will use the **Penguins dataset** from seaborn. It contains measurements of 344 penguins from three species (Adelie, Gentoo, Chinstrap), across three Antarctic islands, with seven features including bill length, bill depth, flipper length, body mass, sex, species, and island. It is richer than the classic Iris dataset for two reasons: it contains both numeric and categorical features, and it has some real missing values that we will need to think about.

### 7.1 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
np.random.seed(42)

# Load the Penguins dataset
df = sns.load_dataset("penguins")
print(f"Dataset shape: {df.shape}")
print(f"\nFirst rows:\n{df.head()}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
```

> **Pedagogical note.** Real datasets almost always have missing values. We will drop the rows with missing values for this lesson to keep things clean, but be aware that imputation strategies (replacing missing values with the mean, median, mode, or a model-based estimate) are a real-world concern.

```python
# Drop rows with missing values for the statistical analysis
df = df.dropna().reset_index(drop=True)
print(f"After dropping missing rows: {df.shape}")
```

### 7.2 Measures of Central Tendency

```python
# Numeric columns we will focus on
numeric_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

print("Means:\n", df[numeric_cols].mean(), "\n")
print("Medians:\n", df[numeric_cols].median(), "\n")
print("Modes:\n", df[numeric_cols].mode().iloc[0])
```

```python
# For the mode of a categorical variable
print(f"\nMost common species: {df['species'].mode()[0]}")
print(f"Most common island: {df['island'].mode()[0]}")
```

```python
# Compare mean and median to spot skewness early
comparison = pd.DataFrame({
    'mean': df[numeric_cols].mean(),
    'median': df[numeric_cols].median(),
})
comparison['mean_minus_median'] = comparison['mean'] - comparison['median']
print(comparison.round(2))
```

The closer "mean − median" is to zero, the more symmetric the distribution. Larger absolute values suggest skewness in one direction or the other.

### 7.3 Measures of Spread

```python
# Range, variance, standard deviation, IQR for each numeric column
spread_df = pd.DataFrame({
    'range': df[numeric_cols].max() - df[numeric_cols].min(),
    'variance': df[numeric_cols].var(),       # ddof=1, sample variance
    'std_dev': df[numeric_cols].std(),         # ddof=1, sample std
    'IQR': df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25),
    'CV (%)': (df[numeric_cols].std() / df[numeric_cols].mean()) * 100,
})
print(spread_df.round(2))
```

Notice the column with the largest standard deviation is `body_mass_g`, but that does not mean it is the "most variable" feature. Body mass is measured in grams of penguins (thousands of grams). The **coefficient of variation** is the column to look at when comparing relative spread.

```python
# Five-number summary
print(df[numeric_cols].describe().loc[['min', '25%', '50%', '75%', 'max']].round(2))
```

```python
# Outlier detection using Tukey's fences on bill_length_mm
col = 'bill_length_mm'
q1 = df[col].quantile(0.25)
q3 = df[col].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = df[(df[col] < lower_fence) | (df[col] > upper_fence)]
print(f"Tukey's fences for {col}: [{lower_fence:.2f}, {upper_fence:.2f}]")
print(f"Number of outliers flagged: {len(outliers)}")
```

### 7.4 Relationships: Covariance and Correlation

```python
# Correlation matrix
corr = df[numeric_cols].corr()
print("Correlation matrix:\n", corr.round(2))
```

```python
# Visualise the correlation matrix
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
plt.title("Penguins, Correlation Matrix of Numeric Features")
plt.show()
```

```python
# The strongest correlation we should expect: flipper length and body mass
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='flipper_length_mm', y='body_mass_g', hue='species', alpha=0.8)
r = df['flipper_length_mm'].corr(df['body_mass_g'])
plt.title(f"Flipper Length vs Body Mass (overall r = {r:.2f})")
plt.show()
```

> **A lesson worth pausing on.** The plot above shows the global correlation between flipper length and body mass is strongly positive, but the relationship is even cleaner *within* each species. This is a hint of a phenomenon called **Simpson's paradox**: relationships at the aggregate level can differ from relationships within subgroups. Always check whether an obvious group structure could be hiding (or exaggerating) the relationship you observe.

```python
# Pairs plot, useful when you have several numeric features
sns.pairplot(df, vars=numeric_cols, hue='species', height=2)
plt.suptitle("Penguins Pairs Plot", y=1.02)
plt.show()
```

### 7.5 Descriptive Statistics: Shape and Visualisation

```python
# Skewness and kurtosis
print("Skewness:\n", df[numeric_cols].skew().round(3))
print("\nKurtosis:\n", df[numeric_cols].kurt().round(3))
```

```python
# Histogram + box plot side by side for body_mass_g
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df['body_mass_g'], bins=25, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Histogram, Body Mass")

sns.boxplot(y=df['body_mass_g'], ax=axes[1], color="seagreen")
axes[1].set_title("Box Plot, Body Mass")

plt.tight_layout()
plt.show()
```

What do you see? The box plot's median line, the box itself (Q1 to Q3), and the whiskers (the non-outlier range) all correspond to the five-number summary you computed in Section 7.3. The histogram, meanwhile, hints at why the distribution looks the way it does: there are heavier and lighter penguins, and that bimodal hint comes from the mix of species.

```python
# Confirm the bimodality by splitting by species
plt.figure(figsize=(9, 4))
for species, group in df.groupby('species'):
    sns.histplot(group['body_mass_g'], bins=20, kde=True, label=species, alpha=0.5)
plt.title("Body Mass by Species")
plt.xlabel("Body Mass (g)")
plt.legend()
plt.show()
```

The aggregate distribution looks somewhat bimodal because it is really three overlapping distributions, one per species. This is the same lesson as Section 7.4: aggregate views can hide group structure.

### 7.6 A Taste of Inference: Simulating the Sampling Distribution

```python
# Treat the full Penguins dataset as our "population"
population = df['flipper_length_mm']
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

# Compare the spread of individual values vs the spread of sample means
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(population, bins=25, kde=True, ax=axes[0], color="steelblue")
axes[0].axvline(true_mean, color='red', linestyle='--', label='Population mean')
axes[0].set_title("Distribution of Individual Flipper Lengths")
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

Two things to notice. First, the sampling distribution of the mean is *narrower* than the original distribution. Sample means are less variable than individual values, and the larger the sample size, the narrower this distribution becomes. Second, the sampling distribution looks more bell-shaped, even though the original flipper length data is somewhat bimodal because of the species mix. This is the **Central Limit Theorem** at work, and it is the reason confidence intervals and hypothesis tests (later this week) are even possible.

## Daily Challenge

Using the Penguins dataset (or a dataset of your choice):

1. Compute the mean, median, and mode for at least two columns. Are the mean and median close together, or do they diverge? What does that suggest about skewness?
2. Calculate the standard deviation and IQR for each numeric column. Which column has the most spread in absolute terms? Which has the most spread in relative terms (CV)?
3. Pick one column and run Tukey's fences. Are there any outliers? If so, what are their species and island? Could the outliers actually be a different species rather than measurement errors?
4. Compute the correlation matrix for the numeric features. Identify the strongest and weakest relationships. For the strongest pair, plot a scatter plot, colour-coded by species. Is the within-species relationship stronger or weaker than the global one?
5. Compute skewness for each column and match each value to what you see in a histogram.
6. Run the sampling distribution simulation from Section 7.6 using two different sample sizes (try `n=5` and `n=50`). How does the spread of the sampling distribution change as sample size increases? Quantify the change.
7. Write 3 to 4 sentences summarising your findings as if explaining them to a non-technical colleague. Make sure to distinguish what your sample shows from what you could (and could not yet) claim about the broader population of all penguins.

> Submit your notebook to the Kanban board under your name by end of day.

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
| Population vs Sample | What you want to know vs what you can measure | (conceptual) |
| Sampling Distribution | The distribution of a statistic across repeated samples | Simulation |

## What's Next

Tomorrow's lesson moves into **distributions**, the mathematical models that describe how data values are spread. We will cover the Uniform, Poisson, and Normal distributions. These distributions are what make the sampling distribution from Section 7.6 behave the way it does, and they form the foundation for everything in Wednesday's probability lesson and Thursday's hypothesis testing.

*Lesson designed for internal use, Statistical Foundations, Data Science Curriculum*
