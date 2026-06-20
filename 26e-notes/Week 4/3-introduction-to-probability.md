# Introduction to Probability

> **Course:** Data Science Foundations
> **Lesson:** Introduction to Probability
> **Prerequisites:** Statistical Foundations, Data Distributions

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Distinguish between probability and likelihood
- Compute marginal, joint, and conditional probabilities
- Understand Bayes' theorem intuitively and apply it to real problems
- Connect probability to the distributions you've already learned
- Solve real-world probability problems in Python

---

## 1. Probability vs. Likelihood — Clarifying the Distinction

These two words are often used interchangeably in everyday language, but in data science they mean very different things.

### Probability

**Probability** asks: *"Given a known model, how likely is this outcome?"*

You know the rules. You're asking about the data.

> **Example:** A coin is fair. What is the probability of getting 3 heads in a row?
> You know the coin is fair (the model is fixed), and you're asking about the outcome.

Probability is always between 0 and 1:
- 0 means the event will never happen
- 1 means the event will always happen
- 0.5 means it's equally likely to happen or not

### Likelihood

**Likelihood** asks: *"Given the data I observed, how plausible is this model?"*

You know the outcome. You're asking about the model.

> **Example:** You flipped a coin 10 times and got 9 heads. How likely is it that this is a fair coin?
> The outcome is fixed (9 heads), and you're evaluating the model (fair vs. biased coin).

### The simple way to remember it

| | Fixed | Unknown |
|---|---|---|
| **Probability** | The model (parameters) | The outcome (data) |
| **Likelihood** | The outcome (data) | The model (parameters) |

This distinction becomes critical in machine learning — model training is fundamentally a likelihood problem.

---

## 2. Marginal Probability

### What it is

Marginal probability is the probability of a **single event occurring**, regardless of any other events. It's the simplest form of probability.

$$P(A)$$

The word "marginal" comes from the practice of summing across the margins of a probability table.

### Example

You survey 200 students about whether they passed their exam and whether they attended extra sessions:

|  | Passed | Failed | **Total** |
|---|---|---|---|
| Attended sessions | 90 | 30 | 120 |
| Did not attend | 40 | 40 | 80 |
| **Total** | 130 | 70 | 200 |

Marginal probabilities:
- P(Passed) = 130 / 200 = **0.65**
- P(Failed) = 70 / 200 = **0.35**
- P(Attended) = 120 / 200 = **0.60**

These are called "marginal" because they only consider one variable at a time, ignoring the other.

---

## 3. Joint Probability

### What it is

Joint probability is the probability of **two events occurring at the same time**.

$$P(A \cap B) = P(A \text{ and } B)$$

### Example

Using the same table above:

- P(Passed **and** Attended) = 90 / 200 = **0.45**
- P(Failed **and** Did not attend) = 40 / 200 = **0.20**

### For independent events

If two events are **independent** (one doesn't affect the other), joint probability simplifies to:

$$P(A \cap B) = P(A) \times P(B)$$

> **Example:** The probability of rolling a 6 on a die AND flipping heads on a coin:
> P(6) × P(Heads) = (1/6) × (1/2) = **1/12 ≈ 0.083**

>  **Important:** This formula only works if the events are truly independent. Exam attendance and passing are NOT independent — that's the whole point of the next section.

---

## 4. Conditional Probability and Bayes' Intuition

### Conditional Probability

Conditional probability is the probability of event A occurring **given that event B has already occurred**.

$$P(A | B) = \frac{P(A \cap B)}{P(B)}$$

Read as: *"The probability of A, given B."*

### Example

What is the probability that a student **passed**, given that they **attended sessions**?

$$P(\text{Passed} | \text{Attended}) = \frac{P(\text{Passed and Attended})}{P(\text{Attended})} = \frac{90/200}{120/200} = \frac{90}{120} = 0.75$$

Compare this to the marginal probability of passing (0.65). Attending sessions increases the probability of passing from 65% to 75% — a meaningful difference.

Conditional probability is how we model **context and new information**.

---

### Bayes' Theorem

Bayes' theorem flips conditional probability — it lets you update your beliefs when new evidence arrives.

$$P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)}$$

In plain English:

> *"How likely is A given B?"* = *"How likely is B given A?"* × *"How likely was A in the first place?"* ÷ *"How likely is B overall?"*

The three components have names:

| Term | Name | Meaning |
|---|---|---|
| P(A) | **Prior** | Your belief about A before seeing any evidence |
| P(B\|A) | **Likelihood** | How probable the evidence is, assuming A is true |
| P(A\|B) | **Posterior** | Your updated belief about A after seeing the evidence |

### Intuitive Example — Medical Testing

A disease affects 1% of the population. A test for it is 95% accurate (true positive rate). If someone tests positive, what is the probability they actually have the disease?

Most people guess ~95%. The real answer is much lower. Let's use Bayes:

- P(Disease) = 0.01 *(prior — 1% of population has it)*
- P(Positive | Disease) = 0.95 *(likelihood — test is 95% accurate)*
- P(Positive) = P(Pos|Disease)×P(Disease) + P(Pos|No Disease)×P(No Disease)
  = 0.95×0.01 + 0.05×0.99 = **0.059**

$$P(\text{Disease} | \text{Positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.161$$

**Only ~16% chance of actually having the disease**, even with a positive test. This is because the disease is so rare that false positives dominate.

This is Bayes' theorem in action — it forces you to account for prior probability, not just test accuracy.

---

## 5. Connecting Distributions Back to Probability

The distributions you learned in the previous lesson are essentially **probability models** — they describe how probability is spread across possible outcomes.

### Discrete distributions → Probability Mass Functions (PMF)

For discrete outcomes (counts, categories), the distribution gives you the exact probability of each value.

> **Poisson example:** If calls arrive at λ=4 per hour, the Poisson distribution tells you:
> - P(exactly 3 calls) = ?
> - P(more than 6 calls) = ?

### Continuous distributions → Probability Density Functions (PDF)

For continuous outcomes (heights, weights, scores), you can't ask for the probability of an exact value — it's effectively zero. Instead, you ask about **ranges**.

> **Normal example:** If exam scores are N(70, 10), what is the probability a student scores between 60 and 80?
> Answer: ~68% — the 68-95-99.7 rule is a probability statement!

### The key connection

Every distribution is a probability tool. Once you know what distribution your data follows, you can make precise probability statements about future observations, anomalies, and expected ranges.

---

## 6. Hands-On: Solving Probability Problems in Python

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

### 6.1 Marginal and Joint Probability from a Table

```python
# Student survey table
data = {
    'Passed': [90, 40],
    'Failed': [30, 40]
}
df = pd.DataFrame(data, index=['Attended', 'Did_not_attend'])
print("Contingency Table:\n", df)

total = df.values.sum()

# Marginal probabilities
p_passed = df['Passed'].sum() / total
p_attended = df.loc['Attended'].sum() / total
print(f"\nP(Passed): {p_passed:.2f}")
print(f"P(Attended): {p_attended:.2f}")

# Joint probability
p_passed_and_attended = df.loc['Attended', 'Passed'] / total
print(f"\nP(Passed and Attended): {p_passed_and_attended:.2f}")
```

---

### 6.2 Conditional Probability

```python
# P(Passed | Attended)
p_passed_given_attended = df.loc['Attended', 'Passed'] / df.loc['Attended'].sum()
print(f"P(Passed | Attended): {p_passed_given_attended:.2f}")

# P(Passed | Did not attend)
p_passed_given_not = df.loc['Did_not_attend', 'Passed'] / df.loc['Did_not_attend'].sum()
print(f"P(Passed | Did not attend): {p_passed_given_not:.2f}")

# Visualise the difference
labels = ['Attended', 'Did not attend']
probs = [p_passed_given_attended, p_passed_given_not]

plt.figure(figsize=(7, 4))
sns.barplot(x=labels, y=probs, palette="muted")
plt.title("P(Passed | Attendance Status)")
plt.ylabel("Probability of Passing")
plt.ylim(0, 1)
plt.show()
```

---

### 6.3 Bayes' Theorem — Medical Test Example

```python
# Known probabilities
p_disease = 0.01               # Prior: 1% prevalence
p_pos_given_disease = 0.95     # True positive rate
p_pos_given_no_disease = 0.05  # False positive rate

# Total probability of a positive test
p_no_disease = 1 - p_disease
p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)

# Bayes: P(Disease | Positive)
p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

print(f"P(Positive): {p_positive:.4f}")
print(f"P(Disease | Positive test): {p_disease_given_positive:.4f} ({p_disease_given_positive*100:.1f}%)")
```

> Try changing `p_disease` to 0.10 (10% prevalence) and observe how dramatically the result changes. This is why screening programs depend heavily on population prevalence.

---

### 6.4 Probability from a Normal Distribution

```python
# Exam scores: N(mean=70, std=10)
mu, sigma = 70, 10
dist = stats.norm(mu, sigma)

# Probability of scoring between 60 and 80
p_60_to_80 = dist.cdf(80) - dist.cdf(60)
print(f"P(60 < score < 80): {p_60_to_80:.4f} ({p_60_to_80*100:.1f}%)")

# Probability of scoring above 90
p_above_90 = 1 - dist.cdf(90)
print(f"P(score > 90): {p_above_90:.4f} ({p_above_90*100:.1f}%)")

# Visualise
x = np.linspace(40, 100, 300)
y = dist.pdf(x)

plt.figure(figsize=(9, 4))
plt.plot(x, y, color='steelblue', lw=2)
plt.fill_between(x, y, where=(x >= 60) & (x <= 80), alpha=0.4, color='steelblue', label='P(60–80)')
plt.fill_between(x, y, where=(x >= 90), alpha=0.4, color='tomato', label='P(>90)')
plt.axvline(mu, color='black', linestyle='--', label='Mean')
plt.title("Normal Distribution — Exam Scores")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.show()
```

---

### 6.5 Probability from a Poisson Distribution

```python
# Support team receives average of 4 tickets per hour
lam = 4
dist_poisson = stats.poisson(lam)

# Exact probability of 6 tickets
p_exactly_6 = dist_poisson.pmf(6)
print(f"P(exactly 6 tickets): {p_exactly_6:.4f}")

# Probability of more than 7 tickets (unusually busy)
p_more_than_7 = 1 - dist_poisson.cdf(7)
print(f"P(more than 7 tickets): {p_more_than_7:.4f}")

# Plot the PMF
k = np.arange(0, 15)
pmf = dist_poisson.pmf(k)

plt.figure(figsize=(8, 4))
plt.bar(k, pmf, color='darkorchid', alpha=0.8)
plt.axvline(lam, color='red', linestyle='--', label=f'λ = {lam}')
plt.title("Poisson Distribution — Support Tickets per Hour")
plt.xlabel("Number of Tickets")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

---

## Daily Challenge

1. A company's data shows that 40% of customers churn. Among churned customers, 70% had contacted support more than 3 times. Among retained customers, only 20% had done so. A new customer has contacted support 4 times — use **Bayes' theorem** to calculate the probability they will churn.

2. Using `scipy.stats.norm`, find the probability that a randomly selected student scores **above 85** in an exam with mean=70 and std=10. Shade this region on a plot.

3. A factory produces defects at an average rate of **2 per batch (λ=2)**. Using the Poisson distribution, calculate:
   - The probability of **zero defects** in a batch
   - The probability of **more than 4 defects**

> Submit your notebook to the Kanban board under your name by end of day.

---

## Summary

| Concept | Question it answers | Key formula |
|---|---|---|
| **Probability** | How likely is this outcome given a known model? | P(A) |
| **Likelihood** | How plausible is this model given the data? | L(model \| data) |
| **Marginal Probability** | How likely is one event on its own? | P(A) |
| **Joint Probability** | How likely are two events together? | P(A ∩ B) |
| **Conditional Probability** | How likely is A, given B already happened? | P(A\|B) = P(A∩B) / P(B) |
| **Bayes' Theorem** | How do I update my belief with new evidence? | P(A\|B) = P(B\|A)·P(A) / P(B) |

---

## What's Next

Tomorrow's lesson moves into **Hypothesis Testing** — using everything you've learned this week (distributions, probability, and the sampling distribution from Monday) to make formal decisions from data. You'll learn how to think about Type I and Type II errors, statistical significance, and how to determine whether a difference you observe is real or just due to chance.

