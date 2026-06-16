# Probability Theory

> **Course:** Data Science Foundations
> **Lesson:** Day 3 — Probability Theory
> **Prerequisites:** Statistics Fundamentals, Distributions

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Distinguish between probability and likelihood
- Compute marginal, joint, and conditional probability
- Understand Bayes' theorem intuitively and apply it to real problems
- Connect probability back to the distributions covered yesterday
- Solve real-world probability problems in Python

---

## 1. Probability vs. Likelihood — Clarifying the Distinction

These two words are often used interchangeably in everyday language, but in data science they mean very different things — and the distinction will matter a great deal once you start training models later in the course.

### 1.1 Probability

**Probability** asks: *"Given a known model, how likely is this outcome?"*

You know the rules. You're asking about the data.

> **Example:** A coin is fair. What is the probability of getting 3 heads in a row?
> The coin's fairness (the model) is fixed and known — you're asking about the outcome.

Probability is always between 0 and 1:
- 0 means the event will never happen
- 1 means the event will always happen
- 0.5 means it's equally likely to happen or not

### 1.2 Likelihood

**Likelihood** asks: *"Given the data I observed, how plausible is this model?"*

You know the outcome. You're asking about the model.

> **Example:** You flip a coin 10 times and get 9 heads. How plausible is it that this is a fair coin?
> The outcome is fixed (9 heads), and you're evaluating the model (fair vs. biased coin).

### 1.3 The Simple Way to Remember It

| | Fixed | Unknown |
|---|---|---|
| **Probability** | The model (parameters) | The outcome (data) |
| **Likelihood** | The outcome (data) | The model (parameters) |

This distinction becomes critical in machine learning — model training is fundamentally a likelihood problem: you hold your observed data fixed and search for the model parameters that make that data most plausible.

---

## 2. Marginal Probability

### 2.1 What It Is

Marginal probability is the probability of a **single event occurring**, regardless of any other events. It's the simplest form of probability.

$$P(A)$$

The word "marginal" comes from the practice of summing across the margins of a probability table — exactly as you'll do below.

### 2.2 Example

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

These are called "marginal" because each one only considers a single variable at a time, ignoring the other entirely — you're reading straight off the row or column totals.

---

## 3. Joint Probability

### 3.1 What It Is

Joint probability is the probability of **two events occurring at the same time**.

$$P(A \cap B) = P(A \text{ and } B)$$

### 3.2 Example

Using the same table above:

- P(Passed **and** Attended) = 90 / 200 = **0.45**
- P(Failed **and** Did not attend) = 40 / 200 = **0.20**

### 3.3 For Independent Events

If two events are **independent** (the occurrence of one has no bearing on the other), joint probability simplifies to a product:

$$P(A \cap B) = P(A) \times P(B)$$

> **Example:** The probability of rolling a 6 on a die *and* flipping heads on a coin:
> P(6) × P(Heads) = (1/6) × (1/2) = **1/12 ≈ 0.083**

> ⚠️ **Important:** This shortcut only works if the events are truly independent. Exam attendance and passing are **not** independent — attending sessions clearly relates to passing — which is exactly why we need conditional probability to handle cases like this properly.

---

## 4. Conditional Probability

### 4.1 What It Is

Conditional probability is the probability of event A occurring **given that event B has already occurred**.

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Read as: *"The probability of A, given B."*

### 4.2 Example

What is the probability that a student **passed**, given that they **attended sessions**?

$$P(\text{Passed} \mid \text{Attended}) = \frac{P(\text{Passed and Attended})}{P(\text{Attended})} = \frac{90/200}{120/200} = \frac{90}{120} = 0.75$$

Compare this to the marginal probability of passing (0.65, from Section 2). Attending sessions raises the probability of passing from 65% to 75% — a meaningful difference, and clear evidence that the two events are **not** independent (which is exactly why the shortcut in Section 3.3 doesn't apply here).

Conditional probability is how we formally model **context and new information** — it's the mathematical tool behind statements like "given what we already know, how should we update our expectations?"

---

## 5. Bayes' Theorem

### 5.1 What It Does

Bayes' theorem flips a conditional probability around — it lets you update your beliefs about something unobservable once new evidence arrives. It is derived directly from the conditional probability formula above:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In plain English:

> *"How likely is A given B?"* = *"How likely is B given A?"* × *"How likely was A in the first place?"* ÷ *"How likely is B overall?"*

### 5.2 The Three Components

| Term | Name | Meaning |
|---|---|---|
| P(A) | **Prior** | Your belief about A before seeing any evidence |
| P(B\|A) | **Likelihood** | How probable the evidence is, assuming A is true |
| P(A\|B) | **Posterior** | Your updated belief about A after seeing the evidence |

Notice that "likelihood" here is exactly the concept introduced in Section 1.2 — Bayes' theorem is, in effect, the formal bridge between probability and likelihood: it tells you precisely how to convert a likelihood into an updated probability, as long as you also account for your prior belief.

### 5.3 Intuitive Example — Medical Testing

A disease affects 1% of the population. A test for it is 95% accurate (true positive rate). If someone tests positive, what is the probability they actually have the disease?

Most people's gut instinct is "about 95%." The real answer is much lower — and working through why is one of the most valuable intuition-builders in this entire lesson.

- P(Disease) = 0.01 *(prior — 1% of the population has it)*
- P(Positive | Disease) = 0.95 *(likelihood — the test is 95% accurate)*
- P(Positive) = P(Pos|Disease)×P(Disease) + P(Pos|No Disease)×P(No Disease)
  = 0.95 × 0.01 + 0.05 × 0.99 = **0.059**

$$P(\text{Disease} \mid \text{Positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.161$$

**Only ~16% chance of actually having the disease**, even with a positive test. Because the disease is so rare to begin with, the much larger pool of healthy people who test falsely positive ends up outnumbering the smaller pool of genuinely sick people who test correctly positive. This is Bayes' theorem in action — it forces you to weigh the **prior probability**, not just the accuracy of the evidence in isolation.

> 💡 This is also why population-level screening programs are evaluated so carefully: a highly accurate test can still produce mostly false alarms when the condition it detects is rare.

---

## 6. Connecting Distributions Back to Probability

Yesterday's distributions — Uniform, Poisson, and Normal — are essentially **probability models**: they describe exactly how probability is spread across all possible outcomes of a variable. Everything you learned about marginal, joint, and conditional probability above can be applied directly to them.

### 6.1 Discrete Distributions → Probability Mass Functions (PMF)

For discrete outcomes (counts, categories), the distribution gives you the exact probability of each value — this is precisely the PMF introduced in yesterday's Section 2.1.

> **Poisson example:** If calls arrive at λ = 4 per hour, the Poisson distribution can tell you:
> - P(exactly 3 calls) = ?
> - P(more than 6 calls) = ?

### 6.2 Continuous Distributions → Probability Density Functions (PDF)

For continuous outcomes (heights, weights, scores), you can't meaningfully ask for the probability of one exact value — as covered yesterday, that probability is effectively zero. Instead, you ask about **ranges**, using the PDF.

> **Normal example:** If exam scores are N(70, 10), what is the probability a student scores between 60 and 80?
> Answer: ~68% — the 68-95-99.7 rule from yesterday's lesson is itself just a probability statement about a range!

### 6.3 The Key Connection

Once you know what distribution your data follows, you can make precise probability statements about future observations, anomalies, and expected ranges — and you can combine that with everything in this lesson (joint, conditional, Bayes') to reason about more complex, multi-variable questions.

---

## 7. Hands-On: Solving Probability Problems in Python

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

### 7.2 Marginal and Joint Probability from a Table

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

### 7.3 Conditional Probability

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

### 7.4 Bayes' Theorem — Medical Test Example

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

> 💡 Try changing `p_disease` to 0.10 (10% prevalence) and observe how dramatically the result changes. This is why screening programs depend so heavily on the prevalence of the condition being tested for, not just the test's accuracy.

---

### 7.5 Probability from a Normal Distribution

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

### 7.6 Probability from a Poisson Distribution

```python
# Support team receives an average of 4 tickets per hour
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

### 7.7 Probability from a Uniform Distribution

```python
# Bus arrives uniformly at random within a 30-minute window
dist_uniform = stats.uniform(loc=0, scale=30)

# Probability of waiting less than 10 minutes
p_wait_under_10 = dist_uniform.cdf(10)
print(f"P(wait < 10 minutes): {p_wait_under_10:.4f}")

# Probability of waiting between 15 and 25 minutes
p_wait_15_to_25 = dist_uniform.cdf(25) - dist_uniform.cdf(15)
print(f"P(15 < wait < 25 minutes): {p_wait_15_to_25:.4f}")
```

> Notice that, unlike the Normal and Poisson examples, every equally-sized interval here has the *same* probability — a direct consequence of yesterday's "flat histogram" shape.

---

## Daily Challenge

1. A company's data shows that 40% of customers churn. Among churned customers, 70% had contacted support more than 3 times. Among retained customers, only 20% had done so. A new customer has contacted support 4 times — use **Bayes' theorem** to calculate the probability they will churn.
2. Using `scipy.stats.norm`, find the probability that a randomly selected student scores **above 85** in an exam with mean=70 and std=10. Shade this region on a plot.
3. A factory produces defects at an average rate of **2 per batch (λ=2)**. Using the Poisson distribution, calculate:
   - The probability of **zero defects** in a batch
   - The probability of **more than 4 defects**
4. Build a 2×2 contingency table from a dataset of your choice (or invent a plausible scenario). Compute all four marginal probabilities, all four joint probabilities, and both relevant conditional probabilities. State in one sentence whether the two variables appear independent.
5. **Bonus:** Revisit the medical testing example in Section 5.3. Holding the test's accuracy fixed at 95%, plot how `P(Disease | Positive)` changes as `p_disease` varies from 0.001 to 0.5. At what prevalence does the posterior probability cross 50%?

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

---

*Lesson designed for internal use — Statistical Foundations, Data Science Curriculum*
