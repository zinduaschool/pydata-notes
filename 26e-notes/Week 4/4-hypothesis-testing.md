# Hypothesis Testing

> **Course:** Data Science Foundations
> **Lesson:** Day 4, Hypothesis Testing
> **Prerequisites:** Statistics Fundamentals, Distributions, Probability Theory

## Learning Objectives

By the end of this lesson, you will be able to:

- Form a null and alternative hypothesis correctly for a real-world question
- Distinguish between Type I and Type II errors, and explain the trade-off between them
- Explain what statistical significance and a p-value actually mean (and, just as importantly, what they do not mean)
- Connect hypothesis testing back to the distributions and probability concepts from earlier this week
- Run a complete hypothesis test end-to-end in Python on both simulated and real data, all the way from raw observations to a business decision

## 1. What is Hypothesis Testing and Where is it Used?

Every day, organisations make decisions like:

- *"Should we change the colour of our sign-up button?"*
- *"Does the new teaching method improve student scores?"*
- *"Is the new drug more effective than the current one?"*
- *"Do customers tip differently for lunch compared to dinner?"*

The naive approach is to make a change and see what happens. The problem? You cannot tell, just by eyeballing a difference, whether the outcome changed *because* of your intervention or just by chance. **Hypothesis testing** is the structured statistical framework for answering that question rigorously, by combining everything covered so far this week. Distributions describe how outcomes vary, probability quantifies how surprising a result is, and hypothesis testing turns that into a formal decision rule.

A common application is **A/B testing**: you split subjects into two groups, a **Control** group (Group A) experiencing the original version, and a **Treatment** group (Group B) experiencing the new version, then use statistics to determine whether any difference you observe between them is real or just random noise.

### 1.1 Where Hypothesis Testing Is Used

- **Product and tech.** Testing new website features, button colours, page layouts (used heavily by Google, Netflix, Meta, and essentially every modern tech company).
- **Marketing.** Comparing email subject lines, ad copy, or campaign formats.
- **Education.** Evaluating whether a new teaching method improves learning outcomes.
- **Healthcare.** Clinical trials comparing a new drug to a placebo, with strict regulatory requirements.
- **Finance.** Testing new pricing strategies or loan approval thresholds.
- **Operations and Quality.** Detecting whether a manufacturing process has drifted out of specification.

Hypothesis testing is one of the most widely used data science tools in industry precisely because it gives a rigorous, defensible answer to "did this change actually work?" rather than a gut-feeling guess. In regulated industries (healthcare, finance), it is often legally required.

## 2. Forming Hypotheses: Null versus Alternative

Before running any test, you must clearly state what you are testing. This is done through two competing statements called **hypotheses**.

### 2.1 The Null Hypothesis (H₀)

The null hypothesis is the **default assumption**. It states that nothing changed, there is no effect, and any difference you observe is purely due to chance.

> *"The new button colour has no effect on sign-up rates."*
> *"The new teaching method produces the same scores as the old one."*
> *"Lunch tips and dinner tips have the same average percentage."*

The null hypothesis is what you are trying to **disprove**. A subtle but important point: you never directly "prove" anything in hypothesis testing. You either find enough evidence to reject H₀, or you do not. "Failing to reject H₀" is not the same as "proving H₀ is true." It just means the evidence is not strong enough to make the call.

### 2.2 The Alternative Hypothesis (H₁ or Hₐ)

The alternative hypothesis is what you believe might be true, the effect you are actually looking for.

> *"The new button colour increases sign-up rates."*
> *"The new teaching method produces higher scores than the old one."*

### 2.3 One-Tailed versus Two-Tailed Tests

| Type | Alternative Hypothesis | Use When |
|---|---|---|
| **Two-tailed** | The groups are different (either direction) | You do not know which direction the effect will go |
| **One-tailed** | Group B is specifically better (or worse) | You have a directional prediction |

> **Rule of thumb.** When in doubt, use a two-tailed test. It is more conservative and less prone to false positives, because it requires evidence of a difference in *either* direction rather than committing in advance to one direction.

A critical caveat on one-tailed tests: you must commit to the direction *before* looking at the data. Picking the direction after seeing the result is a form of cheating that inflates your false positive rate dramatically. This is called "p-hacking" and it has caused real harm in scientific literature.

### 2.4 A Concrete Example

You run an online course platform. Your current landing page converts at 10%. You redesign it and want to know if the new design performs differently.

- **H₀:** Conversion rate of the new page = conversion rate of the old page
- **H₁:** Conversion rate of the new page ≠ conversion rate of the old page (two-tailed)

## 3. Errors in Hypothesis Testing

Every hypothesis test is a decision made under uncertainty, and like any such decision, it can go wrong in two distinct ways. Understanding both is essential, because reducing one type of error often increases the other.

### 3.1 The Two Types of Error

| | H₀ is actually True | H₀ is actually False |
|---|---|---|
| **You reject H₀** | **Type I Error** (False Positive) | Correct decision |
| **You fail to reject H₀** | Correct decision | **Type II Error** (False Negative) |

### 3.2 Type I Error (False Positive)

A **Type I error** occurs when you reject the null hypothesis even though it was actually true. You conclude there is an effect when there is not one.

> **Example.** You conclude the new landing page increases conversions, ship it company-wide, and it turns out the original "improvement" was just random noise in your sample.

The probability of making a Type I error is exactly the **significance level, α**, that you set before running the test (covered in Section 4). This is precisely *why* α represents your tolerance for being wrong: it is a number you choose, not something you measure after the fact.

### 3.3 Type II Error (False Negative)

A **Type II error** occurs when you fail to reject the null hypothesis even though it was actually false. You conclude there is no effect when there really is one.

> **Example.** The new teaching method genuinely does improve scores, but your sample was too small or too noisy to detect it, so you incorrectly conclude "no significant difference" and abandon a method that actually works.

The probability of a Type II error is denoted **β**. Its counterpart, **1 − β**, is called **statistical power**, the probability of correctly detecting a real effect when one exists. The conventional target for power is 0.80 or higher, meaning you want at least an 80% chance of catching a real effect of a meaningful size.

### 3.4 The Trade-Off

Type I and Type II errors are in tension with each other. If you make your significance threshold stricter (lower α, say 0.01 instead of 0.05) to reduce false positives, you simultaneously make it *harder* to detect real effects, raising the risk of a Type II error, unless you compensate with a larger sample size. There is no way to drive both error rates to zero simultaneously with a fixed sample size. The only way to reduce both at once is to collect more data.

### 3.5 Choosing Based on Real-World Cost

Which error matters more depends entirely on the context, and is a judgment call that has to be made *before* the data comes in:

| Scenario | Worse Error | Why |
|---|---|---|
| Medical diagnosis screening | Type II (missing a real disease) | A false negative could mean a patient is not treated |
| Criminal trial | Type I (convicting an innocent person) | The justice system is designed to weight this heavily |
| Spam filtering | Type I (blocking a legitimate email) | Users are often more frustrated by lost mail than by occasional spam |
| New drug approval | Type I (approving an ineffective or harmful drug) | Regulatory bodies set very strict significance thresholds |
| Catching production bugs | Type II (missing a real regression) | Letting bad code reach users is worse than rolling back a working feature |

> **A pedagogical habit worth forming.** Whenever you set up a hypothesis test, explicitly ask: "In this situation, which mistake would be more costly, a false alarm or a missed effect?" That answer should directly inform how strict your significance level needs to be.

## 4. Statistical Significance and P-Values

### 4.1 The Core Question

After running your test, you will observe some difference between Group A and Group B. But is that difference meaningful, or could it have happened by chance, even if there was no real effect? That is exactly what **statistical significance** and **p-values** help answer.

### 4.2 What is a P-Value?

The **p-value** is the probability of observing a difference as large as (or larger than) the one you measured, **assuming the null hypothesis is true**.

In simpler terms:

> *"If there were actually no effect, how likely is it that I would see results this extreme just by chance?"*

- A **small p-value** means this result would be very unlikely by chance, which is evidence against H₀
- A **large p-value** means this result could easily happen by chance, so there is not enough evidence to reject H₀

### 4.3 The Significance Threshold (α)

Before running a test, you set a **significance level, alpha (α)**, which represents your tolerance for making a Type I error. As established in Section 3.2, this is a number you commit to in advance, not a value you tune after seeing the result.

The most common choice is **α = 0.05**, meaning you accept a 5% chance of incorrectly rejecting H₀ when it is actually true. In high-stakes scientific work, much stricter thresholds (0.01 or even 0.001) are sometimes used.

| Result | Decision |
|---|---|
| p-value ≤ α (e.g. ≤ 0.05) | **Reject H₀**: the result is statistically significant |
| p-value > α (e.g. > 0.05) | **Fail to reject H₀**: not enough evidence |

> **Common misconception.** A p-value is **NOT** the probability that H₀ is true. It is the probability of your data (or more extreme data) *given that H₀ is true*. This is a subtle but critical distinction. The p-value says nothing directly about how likely your hypothesis is. It only describes how surprising your data would be under the null. Misinterpreting p-values as posterior probabilities is one of the most common and consequential errors in applied statistics.

### 4.4 Statistical Significance Does Not Mean Practical Significance

A result can be **statistically significant** but still **practically meaningless**.

> **Example.** You test a new landing page on 1 million users. The new page converts at 10.01% versus 10.00%. With that sample size, this tiny difference might well be statistically significant. But would you rebuild your entire site for a 0.01% lift?

Always ask: *"Is this difference big enough to matter in practice?"* alongside *"Is it statistically significant?"* This is the same caution from Monday's correlation discussion, applied to differences between groups instead of relationships between variables. A small effect, given enough data, can always be made to look statistically significant. **Effect size** measures (Cohen's d, relative lift, odds ratios) are how you quantify practical significance separately from p-values.

## 5. Connecting Back to Distributions and Probability

Hypothesis testing is built entirely on the concepts from the previous three lessons. Here is exactly how they connect.

### 5.1 The Test Statistic

When you compare two groups, you compute a **test statistic**, a single number summarising how different the groups are, relative to the variability in the data. For comparing means, the most common is the **t-statistic**:

$$t = \frac{\bar{x}_A - \bar{x}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$

Where:

- $\bar{x}_A$, $\bar{x}_B$ are the group means (Section 2 of Monday)
- $s_A^2$, $s_B^2$ are the group variances (Section 3 of Monday)
- $n_A$, $n_B$ are the group sample sizes

The numerator is the difference between groups. The denominator is essentially the combined standard error. If the difference is large relative to the noise, the t-statistic is large in absolute value, and the data is hard to explain under H₀.

### 5.2 The Sampling Distribution

Recall the **sampling distribution of the mean** from Monday's lesson, and the Central Limit Theorem behind it. Under the null hypothesis, if you repeated your experiment many times, the test statistic itself would follow a known distribution. Typically this is the **t-distribution**, or the **Normal distribution** for large samples (Tuesday's lesson). This is precisely why Tuesday spent so much time on the shape and behaviour of the Normal distribution: that shape is what makes the next step possible.

### 5.3 Where the P-Value Comes From

The p-value is the **area in the tail(s)** of that sampling distribution beyond your observed test statistic, exactly the same kind of area-under-the-curve calculation as the continuous probability questions from Wednesday's lesson.

- If your test statistic lands far out in the tail, then the p-value is small, and the result is unlikely under H₀
- If it lands near the centre, then the p-value is large, and the result is consistent with H₀

In other words, a p-value answers the exact same kind of question you practiced in Wednesday's lesson, *"what is the probability of observing this value or something more extreme, given this distribution?"*, just applied to a test statistic instead of a raw measurement.

## 6. Hands-On Part 1: A/B Test on Simulated Data

We will start with a simulated A/B test so the mechanics are clear, then move to a real dataset in Part 2.

### 6.1 Scenario

You run an e-learning platform. You want to test whether a **new course landing page** leads to higher sign-up rates than the current one.

- **Group A (Control):** Current landing page
- **Group B (Treatment):** New landing page
- **Metric:** Did the user sign up? (1 = yes, 0 = no)
- **Significance level:** α = 0.05

### 6.2 Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(style="whitegrid")
np.random.seed(42)
```

### 6.3 Simulate the Data

```python
# Group A: 1000 users, 10% conversion rate
n_A = 1000
conv_rate_A = 0.10
group_A = np.random.binomial(1, conv_rate_A, n_A)

# Group B: 1000 users, 12% conversion rate (the "true" improvement)
n_B = 1000
conv_rate_B = 0.12
group_B = np.random.binomial(1, conv_rate_B, n_B)

print(f"Group A, Conversions: {group_A.sum()} / {n_A} ({group_A.mean()*100:.1f}%)")
print(f"Group B, Conversions: {group_B.sum()} / {n_B} ({group_B.mean()*100:.1f}%)")
```

### 6.4 State the Hypotheses

```python
alpha = 0.05
print("Null Hypothesis (H0): No difference between Group A and Group B")
print("Alternative Hypothesis (H1): There is a difference (two-tailed)")
print(f"Significance level: α = {alpha}")
```

### 6.5 Run the Statistical Test

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

# Z-statistic and two-tailed p-value
z_stat = (p_B - p_A) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"\nGroup A conversion rate: {p_A:.4f}")
print(f"Group B conversion rate: {p_B:.4f}")
print(f"Observed difference: {(p_B - p_A)*100:.2f} percentage points")
print(f"\nZ-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.4f}")
```

### 6.6 Interpret the Result

```python
if p_value <= alpha:
    print(f"\nResult: Statistically significant (p={p_value:.4f} ≤ α={alpha})")
    print("Decision: Reject H0, the new landing page performs differently.")
else:
    print(f"\nResult: Not statistically significant (p={p_value:.4f} > α={alpha})")
    print("Decision: Fail to reject H0, no strong evidence of a difference.")
```

### 6.7 Visualise the Result

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Conversion rates comparison
groups = ['Group A\n(Control)', 'Group B\n(Treatment)']
rates = [p_A * 100, p_B * 100]
colors = ['steelblue', 'seagreen']

axes[0].bar(groups, rates, color=colors, alpha=0.85, width=0.4)
axes[0].set_title("Conversion Rates: A vs B", fontsize=13)
axes[0].set_ylabel("Conversion Rate (%)")
axes[0].set_ylim(0, max(rates) * 1.3)
for i, v in enumerate(rates):
    axes[0].text(i, v + 0.2, f"{v:.1f}%", ha='center', fontweight='bold')

# Plot 2: Z-distribution and p-value region
x = np.linspace(-4, 4, 300)
y = stats.norm.pdf(x)

axes[1].plot(x, y, color='black', lw=2)
axes[1].fill_between(x, y, where=(x >= abs(z_stat)), color='tomato', alpha=0.5, label='p-value region')
axes[1].fill_between(x, y, where=(x <= -abs(z_stat)), color='tomato', alpha=0.5)
axes[1].axvline(z_stat, color='tomato', linestyle='--', label=f'z = {z_stat:.2f}')
axes[1].axvline(-z_stat, color='tomato', linestyle='--')
axes[1].set_title("Z-Distribution and P-value", fontsize=13)
axes[1].set_xlabel("Z-statistic")
axes[1].set_ylabel("Density")
axes[1].legend()

plt.suptitle(f"A/B Test Results, p-value = {p_value:.4f}, α = {alpha}", fontsize=12, y=1.02)
plt.tight_layout()
plt.show()
```

### 6.8 Practical Significance

```python
absolute_lift = p_B - p_A
relative_lift = (p_B - p_A) / p_A * 100

print(f"Absolute lift: {absolute_lift*100:.2f} percentage points")
print(f"Relative lift: {relative_lift:.1f}%")
print(f"\nWith 10,000 monthly visitors, the new page would generate")
print(f"roughly {int(absolute_lift * 10000)} additional sign-ups per month.")
```

This final step is what turns a statistical result into a business decision. A 2 percentage point lift might be worth shipping, or it might not, depending on the cost of the redesign.

## 7. Hands-On Part 2: A Real A/B Question on the Tips Dataset

Real-world hypothesis testing rarely involves cleanly simulated binary outcomes. To practice on more realistic data, we will use the **Tips dataset** from seaborn, which records 244 restaurant bills with the total bill, tip amount, day of the week, time (lunch or dinner), party size, gender of the bill payer, and whether the party included smokers. This is a perfect dataset for asking real A/B-style questions about human behaviour.

### 7.1 Load the Data

```python
tips = sns.load_dataset("tips")
print(tips.head())
print(f"\nShape: {tips.shape}")

# Compute tip percentage for cleaner analysis
tips['tip_pct'] = tips['tip'] / tips['total_bill'] * 100
print(f"\nMean tip percentage: {tips['tip_pct'].mean():.2f}%")
```

### 7.2 Question 1, Do Smokers and Non-Smokers Tip Differently?

This is a real, comparable, two-group question. Both groups have continuous (numeric) outcomes, so we use a **two-sample t-test**.

```python
# Hypotheses
print("H0: Mean tip percentage is the same for smokers and non-smokers")
print("H1: Mean tip percentage differs between smokers and non-smokers (two-tailed)")

# Split and summarise
smoker_tips = tips[tips['smoker'] == 'Yes']['tip_pct']
nonsmoker_tips = tips[tips['smoker'] == 'No']['tip_pct']

print(f"\nSmokers (n={len(smoker_tips)}):     mean = {smoker_tips.mean():.2f}%, std = {smoker_tips.std():.2f}")
print(f"Non-smokers (n={len(nonsmoker_tips)}): mean = {nonsmoker_tips.mean():.2f}%, std = {nonsmoker_tips.std():.2f}")
```

```python
# Two-sample t-test (independent samples)
t_stat, p_value = stats.ttest_ind(smoker_tips, nonsmoker_tips)
alpha = 0.05

print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value:     {p_value:.4f}")

if p_value <= alpha:
    print(f"\nResult: Reject H0. Smokers and non-smokers tip at different rates.")
else:
    print(f"\nResult: Fail to reject H0. No strong evidence of a tipping difference.")
```

```python
# Visualise
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sns.boxplot(data=tips, x='smoker', y='tip_pct', ax=axes[0], palette='muted')
axes[0].set_title("Tip Percentage by Smoker Status")
axes[0].set_ylabel("Tip Percentage")

sns.histplot(data=tips, x='tip_pct', hue='smoker', kde=True, ax=axes[1], alpha=0.5)
axes[1].set_title("Tip Percentage Distributions")

plt.tight_layout()
plt.show()
```

### 7.3 Question 2, Are Lunch and Dinner Tips Different?

```python
print("H0: Mean tip percentage is the same for lunch and dinner")
print("H1: Mean tip percentage differs between lunch and dinner (two-tailed)")

lunch_tips = tips[tips['time'] == 'Lunch']['tip_pct']
dinner_tips = tips[tips['time'] == 'Dinner']['tip_pct']

print(f"\nLunch (n={len(lunch_tips)}):  mean = {lunch_tips.mean():.2f}%, std = {lunch_tips.std():.2f}")
print(f"Dinner (n={len(dinner_tips)}): mean = {dinner_tips.mean():.2f}%, std = {dinner_tips.std():.2f}")

t_stat_2, p_value_2 = stats.ttest_ind(lunch_tips, dinner_tips)
print(f"\nt-statistic: {t_stat_2:.4f}")
print(f"p-value:     {p_value_2:.4f}")
```

### 7.4 Question 3, Is Smoker Status Independent of Day?

When both variables are categorical, we use a **chi-square test of independence** instead of a t-test.

```python
print("H0: Smoker status is independent of day")
print("H1: Smoker status and day are associated")

contingency = pd.crosstab(tips['smoker'], tips['day'])
print("\nContingency table:\n", contingency)

chi2, p_value_3, dof, expected = stats.chi2_contingency(contingency)
print(f"\nChi-square statistic: {chi2:.4f}")
print(f"Degrees of freedom:   {dof}")
print(f"p-value:              {p_value_3:.4f}")

if p_value_3 <= 0.05:
    print("\nResult: Reject H0. Smoker status varies by day.")
else:
    print("\nResult: Fail to reject H0. No clear association.")
```

## 8. Hands-On Part 3: Simulating Type I and Type II Errors Directly

It is one thing to define Type I and Type II errors on paper. It is far more convincing to actually generate them by repeating an experiment many times.

```python
# Simulating Type I error rate
# Run many "experiments" where there is truly NO difference between groups (H0 is true)
# and count how often we incorrectly reject H0 at alpha = 0.05

n_simulations = 2000
alpha = 0.05
false_positives = 0

for _ in range(n_simulations):
    sim_A = np.random.binomial(1, 0.10, 1000)  # both groups have the SAME true rate
    sim_B = np.random.binomial(1, 0.10, 1000)

    pa, pb = sim_A.mean(), sim_B.mean()
    pooled = (sim_A.sum() + sim_B.sum()) / 2000
    se_sim = np.sqrt(pooled * (1 - pooled) * (2/1000))
    z = (pb - pa) / se_sim if se_sim > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    if p_val <= alpha:
        false_positives += 1

print(f"Observed Type I error rate: {false_positives/n_simulations:.4f}")
print(f"Expected (by definition, this should be close to alpha): {alpha}")
```

```python
# Simulating Type II error rate (and statistical power)
# Run many experiments where there IS a true difference (H1 is actually true)
# and count how often we FAIL to detect it

true_diffs_missed = 0

for _ in range(n_simulations):
    sim_A = np.random.binomial(1, 0.10, 1000)  # true rate: 10%
    sim_B = np.random.binomial(1, 0.12, 1000)  # true rate: 12% (a real effect exists)

    pa, pb = sim_A.mean(), sim_B.mean()
    pooled = (sim_A.sum() + sim_B.sum()) / 2000
    se_sim = np.sqrt(pooled * (1 - pooled) * (2/1000))
    z = (pb - pa) / se_sim if se_sim > 0 else 0
    p_val = 2 * (1 - stats.norm.cdf(abs(z)))

    if p_val > alpha:  # we failed to detect a real effect
        true_diffs_missed += 1

power = 1 - (true_diffs_missed / n_simulations)
print(f"Observed Type II error rate (beta): {true_diffs_missed/n_simulations:.4f}")
print(f"Observed statistical power (1 - beta): {power:.4f}")
```

> **What to notice.** The simulated Type I error rate should land very close to `alpha = 0.05`, confirming that α really is exactly the false-positive rate you signed up for. The Type II error rate, by contrast, depends heavily on the size of the true effect and the sample size. Try lowering `n` from 1000 to 200 in the second simulation and watch the power drop substantially. This is the practical meaning of "underpowered study," and it is the most common reason real research fails to detect real effects.

## Daily Challenge

You work at an e-learning company. Two versions of a course thumbnail were shown to users:

- **Version A:** Text-heavy thumbnail, 180 clicks out of 2,000 impressions
- **Version B:** Image-led thumbnail, 230 clicks out of 2,000 impressions

1. State your null and alternative hypotheses.
2. Run a two-proportion z-test in Python.
3. At α = 0.05, what is your decision?
4. Calculate the absolute and relative lift.
5. In this specific scenario, would a Type I or a Type II error be more costly for the business? Justify your answer in 2 to 3 sentences.
6. Write 3 to 4 sentences explaining your findings to a non-technical stakeholder, being careful to distinguish statistical significance from practical significance.
7. Using the Tips dataset from Section 7, run a t-test comparing tip percentages between male and female bill payers. Report the result and interpret it.
8. **Bonus.** Modify the Section 8 simulation so that Group A and Group B both have a true conversion rate of 12% instead of 10% versus 12%. Re-run the Type I error simulation with this new shared rate and confirm the false-positive rate still hovers around α.

> Submit your notebook to the Kanban board under your name by end of day.

## Summary

| Concept | What It Means |
|---|---|
| **Hypothesis Test / A/B Test** | A controlled comparison used to determine if an observed difference is real |
| **Null Hypothesis (H₀)** | The default assumption, no difference exists |
| **Alternative Hypothesis (H₁)** | The claim being tested, a difference exists |
| **Type I Error (α)** | Rejecting H₀ when it is actually true (false positive) |
| **Type II Error (β)** | Failing to reject H₀ when it is actually false (false negative) |
| **Statistical Power (1 − β)** | The probability of correctly detecting a real effect |
| **P-value** | Probability of observing your result (or more extreme) if H₀ were true |
| **Significance level (α)** | Your pre-chosen threshold for rejecting H₀, typically 0.05 |
| **Statistical significance** | p ≤ α implies reject H₀ |
| **Practical significance** | Is the difference large enough to matter in the real world? |

## Putting It All Together: The Statistics and Probability Journey

You have now completed the statistics and probability portion of this module. Here is how every lesson connects:

```
Mon: Stats Fundamentals -> Tue: Distributions -> Wed: Probability -> Thu: Hypothesis Testing
        |                       |                    |                     |
   Summarise data          Model the data       Quantify              Make decisions
   (mean, std,              (Uniform,            uncertainty           from data
    variance,                Poisson,             (joint,               (errors,
    correlation)             Normal)              conditional,          significance,
                                                  Bayes)                p-values)
```

Every tool in data science, from machine learning to dashboards, builds on this foundation. The remaining two days of this module (Friday and Saturday) shift gears toward the **mathematics** underneath those models: linear algebra and calculus, which power everything from the normal equations behind linear regression to the gradient descent that trains most modern machine learning algorithms.

*Lesson designed for internal use, Statistical Foundations, Data Science Curriculum*
