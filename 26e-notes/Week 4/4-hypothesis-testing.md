# Hypothesis Testing & A/B Testing

> **Course:** Data Science Foundations
> **Lesson:** Hypothesis Testing
> **Prerequisites:** Statistical Foundations, Data Distributions, Introduction to Probability

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what A/B testing is and where it is applied in the real world
- Form a null and alternative hypothesis correctly
- Interpret p-values and statistical significance
- Connect hypothesis testing back to distributions and probability
- Run a complete A/B test end-to-end in Python

---

## 1. What is A/B Testing and Where is it Used?

Every day, companies make decisions like:

- *"Should we change the colour of our sign-up button?"*
- *"Does the new teaching method improve student scores?"*
- *"Is the new drug more effective than the current one?"*

The naive approach is to make a change and see what happens. The problem? You can't tell if the outcome changed because of your intervention — or just by chance.

**A/B testing** is a structured way to answer these questions. You split your subjects into two groups:

- **Group A (Control):** Experiences the original version
- **Group B (Treatment):** Experiences the new version

You then measure a specific outcome for both groups and use statistics to determine whether any difference you observe is **real** or just **random noise**.

### Where it is used

- **Product & tech:** Testing new website features, button colours, page layouts (used heavily by companies like Google, Netflix, and Meta)
- **Marketing:** Comparing email subject lines, ad copy, or campaign formats
- **Education:** Evaluating whether a new teaching method improves learning outcomes
- **Healthcare:** Clinical trials comparing a new drug to a placebo
- **Finance:** Testing new pricing strategies or loan approval thresholds

A/B testing is one of the most widely used data science tools in industry precisely because it gives you a rigorous, defensible answer to *"did this change actually work?"*

---

## 2. Forming Hypotheses — Null vs. Alternative

Before you run any test, you must clearly state what you are testing. This is done through two competing statements called **hypotheses**.

### The Null Hypothesis (H₀)

The null hypothesis is the **default assumption** — it states that nothing changed, there is no effect, and any difference you observe is due to chance.

> *"The new button colour has no effect on sign-up rates."*
> *"The new teaching method produces the same scores as the old one."*

The null hypothesis is what you are trying to **disprove**.

### The Alternative Hypothesis (H₁ or Hₐ)

The alternative hypothesis is what you believe might be true — the effect you are looking for.

> *"The new button colour increases sign-up rates."*
> *"The new teaching method produces higher scores than the old one."*

### One-tailed vs. Two-tailed tests

| Type | Alternative Hypothesis | Use when |
|---|---|---|
| **Two-tailed** | The groups are different (either direction) | You don't know which direction the effect will go |
| **One-tailed** | Group B is specifically better (or worse) | You have a directional prediction |

> **Rule of thumb:** When in doubt, use a two-tailed test. It's more conservative and less prone to false positives.

### A concrete example

You run an online course platform. Your current landing page converts at 10%. You redesign it and want to know if the new design performs differently.

- **H₀:** Conversion rate of new page = Conversion rate of old page
- **H₁:** Conversion rate of new page ≠ Conversion rate of old page *(two-tailed)*

---

## 3. Statistical Significance and P-Values

### The core question

After running your test, you'll observe a difference between Group A and Group B. But is that difference meaningful — or could it have happened by chance, even if there was no real effect?

That's exactly what **statistical significance** and **p-values** help you answer.

### What is a p-value?

The **p-value** is the probability of observing a difference as large as (or larger than) the one you measured, **assuming the null hypothesis is true**.

In simpler terms:

> *"If there were actually no effect, how likely is it that I'd see results this extreme just by chance?"*

- A **small p-value** means: this result would be very unlikely by chance → evidence against H₀
- A **large p-value** means: this result could easily happen by chance → not enough evidence to reject H₀

### The significance threshold (α)

Before running a test, you set a **significance level** — called **alpha (α)** — which is your tolerance for being wrong.

The most common choice is **α = 0.05**, meaning you accept a 5% chance of incorrectly rejecting H₀.

| Result | Decision |
|---|---|
| p-value ≤ α (e.g. ≤ 0.05) | **Reject H₀** — the result is statistically significant |
| p-value > α (e.g. > 0.05) | **Fail to reject H₀** — not enough evidence |

>  **Common misconception:** A p-value is NOT the probability that H₀ is true. It is the probability of your data (or more extreme data) given that H₀ is true. This is a subtle but important distinction.

### Statistical significance ≠ Practical significance

A result can be **statistically significant** but still **practically meaningless**.

> **Example:** You test a new landing page on 1 million users. The new page converts at 10.01% vs. 10.00%. With that sample size, this tiny difference might be statistically significant — but would you rebuild your entire site for a 0.01% lift?

Always ask: *"Is this difference big enough to matter in practice?"* alongside *"Is it statistically significant?"*

---

## 4. Connecting Back to Distributions and Probability

Hypothesis testing is built entirely on the concepts from the previous lessons. Here's how they connect:

### The test statistic

When you compare two groups, you compute a **test statistic** — a single number that summarises how different the groups are, relative to the variability in the data.

For comparing means, the most common is the **t-statistic**:

$$t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$

Where:
- $\bar{x}_A$, $\bar{x}_B$ are the group means
- $s_A^2$, $s_B^2$ are the group variances
- $n_A$, $n_B$ are the group sizes

### The sampling distribution

Under the null hypothesis, if you repeated your experiment many times, the test statistic would follow a known distribution — typically the **t-distribution** or **normal distribution** for large samples.

### Where the p-value comes from

The p-value is the **area in the tail(s)** of that distribution beyond your observed test statistic.

- If your t-statistic lands far out in the tail → small p-value → unlikely under H₀
- If it lands near the center → large p-value → consistent with H₀

This is exactly the same logic as the probability questions you solved in the last lesson — you're asking: *"What is the probability of observing this value or more extreme, given this distribution?"*

---

## 5. Hands-On: Running a Simple A/B Test End-to-End in Python

### Scenario

You run an e-learning platform. You want to test whether a **new course landing page** leads to higher sign-up rates than the current one.

- **Group A (Control):** Current landing page
- **Group B (Treatment):** New landing page
- **Metric:** Did the user sign up? (1 = yes, 0 = no)
- **Significance level:** α = 0.05

---

### Setup

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

### Step 1: Simulate the Data

```python
# Group A: 1000 users, 10% conversion rate
n_A = 1000
conv_rate_A = 0.10
group_A = np.random.binomial(1, conv_rate_A, n_A)

# Group B: 1000 users, 12% conversion rate (the "true" improvement)
n_B = 1000
conv_rate_B = 0.12
group_B = np.random.binomial(1, conv_rate_B, n_B)

print(f"Group A — Conversions: {group_A.sum()} / {n_A} ({group_A.mean()*100:.1f}%)")
print(f"Group B — Conversions: {group_B.sum()} / {n_B} ({group_B.mean()*100:.1f}%)")
```

---

### Step 2: State the Hypotheses

```python
# H0: Conversion rate of A == Conversion rate of B
# H1: Conversion rate of A != Conversion rate of B (two-tailed)
# Significance level
alpha = 0.05
print(f"Null Hypothesis (H0): No difference between Group A and Group B")
print(f"Alternative Hypothesis (H1): There is a difference")
print(f"Significance level: α = {alpha}")
```

---

### Step 3: Run the Statistical Test

For conversion rates (binary outcomes), we use a **two-proportion z-test**.

```python
# Observed conversions and totals
conv_A = group_A.sum()
conv_B = group_B.sum()

# Proportions
p_A = conv_A / n_A
p_B = conv_B / n_B

# Pooled proportion under H0
p_pool = (conv_A + conv_B) / (n_A + n_B)

# Standard error
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))

# Z-statistic
z_stat = (p_B - p_A) / se

# Two-tailed p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nGroup A conversion rate: {p_A:.4f}")
print(f"Group B conversion rate: {p_B:.4f}")
print(f"Observed difference: {(p_B - p_A)*100:.2f} percentage points")
print(f"\nZ-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

---

### Step 4: Interpret the Result

```python
if p_value <= alpha:
    print(f"\n Result: Statistically significant (p={p_value:.4f} ≤ α={alpha})")
    print("Decision: Reject H0 — the new landing page performs differently.")
else:
    print(f"\n Result: Not statistically significant (p={p_value:.4f} > α={alpha})")
    print("Decision: Fail to reject H0 — no strong evidence of a difference.")
```

---

### Step 5: Visualise the Results

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Plot 1: Conversion rates comparison ---
groups = ['Group A\n(Control)', 'Group B\n(Treatment)']
rates = [p_A * 100, p_B * 100]
colors = ['steelblue', 'seagreen']

axes[0].bar(groups, rates, color=colors, alpha=0.85, width=0.4)
axes[0].set_title("Conversion Rates: A vs. B", fontsize=13)
axes[0].set_ylabel("Conversion Rate (%)")
axes[0].set_ylim(0, max(rates) * 1.3)
for i, v in enumerate(rates):
    axes[0].text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

# --- Plot 2: Z-distribution and p-value ---
x = np.linspace(-4, 4, 300)
y = stats.norm.pdf(x)

axes[1].plot(x, y, color='black', lw=2)
axes[1].fill_between(x, y, where=(x >= abs(z_stat)), color='tomato', alpha=0.5, label=f'p-value region')
axes[1].fill_between(x, y, where=(x <= -abs(z_stat)), color='tomato', alpha=0.5)
axes[1].axvline(z_stat, color='tomato', linestyle='--', label=f'z = {z_stat:.2f}')
axes[1].axvline(-z_stat, color='tomato', linestyle='--')
axes[1].set_title("Z-Distribution and P-value", fontsize=13)
axes[1].set_xlabel("Z-statistic")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.suptitle(f"A/B Test Results  |  p-value = {p_value:.4f}  |  α = {alpha}", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
```

---

### Step 6: Check Practical Significance

```python
# Absolute and relative lift
absolute_lift = p_B - p_A
relative_lift = (p_B - p_A) / p_A * 100

print(f"Absolute lift: {absolute_lift*100:.2f} percentage points")
print(f"Relative lift: {relative_lift:.1f}%")
print(f"\nIf we have 10,000 monthly visitors, the new page would generate")
print(f"an additional ~{int(absolute_lift * 10000)} sign-ups per month.")
```

>  This final step is what turns a statistical result into a **business decision**. A 2 percentage point lift might be worth shipping — or it might not, depending on the cost of the redesign.

---

## Daily Challenge

You work at an e-learning company. Two versions of a course thumbnail were shown to users:

- **Version A:** Text-heavy thumbnail → 180 clicks out of 2,000 impressions
- **Version B:** Image-led thumbnail → 230 clicks out of 2,000 impressions

1. State your null and alternative hypotheses
2. Run a two-proportion z-test in Python
3. At α = 0.05, what is your decision?
4. Calculate the absolute and relative lift
5. Write 3–4 sentences explaining your findings to a non-technical stakeholder

> Submit your notebook to the Kanban board under your name by end of day.

---

## Summary

| Concept | What it means |
|---|---|
| **A/B Test** | A controlled experiment comparing two versions of something |
| **Null Hypothesis (H₀)** | The default assumption — no difference exists |
| **Alternative Hypothesis (H₁)** | The claim you're testing — a difference exists |
| **P-value** | Probability of observing your result (or more extreme) if H₀ were true |
| **Significance level (α)** | Your threshold for rejecting H₀ — typically 0.05 |
| **Statistical significance** | p ≤ α → reject H₀ |
| **Practical significance** | Is the difference large enough to matter in the real world? |

---

## Putting It All Together — The Statistical Foundations Journey

You've now completed the Statistical Foundations module. Here's how every lesson connects:

```
Descriptive Stats → Distributions → Probability → Hypothesis Testing
      ↓                  ↓               ↓                ↓
 Summarise data     Model the data    Quantify           Make
  (mean, std,       (normal,          uncertainty        decisions
  variance)         Poisson, etc.)    (p-values,         from data
                                      Bayes)
```

Every tool in data science — from machine learning to dashboards — builds on this foundation.

---

*Lesson designed for internal use — Statistical Foundations, Data Science Curriculum*