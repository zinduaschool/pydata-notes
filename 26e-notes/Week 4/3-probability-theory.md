# Probability Theory

> **Course:** Data Science Foundations
> **Lesson:** Day 3, Probability Theory
> **Prerequisites:** Statistics Fundamentals, Distributions

## Learning Objectives

By the end of this lesson, you will be able to:

- Distinguish between probability and likelihood, two terms that are often used interchangeably but mean very different things in data science
- Compute marginal, joint, and conditional probability from contingency tables and from problem statements
- Understand Bayes' theorem intuitively and apply it to real problems involving rare events
- Connect probability back to the distributions covered yesterday, in particular by computing probabilities from Normal and Poisson distributions
- Solve real-world probability problems in Python using contingency tables and `scipy.stats`

## 1. Probability versus Likelihood: Clarifying the Distinction

These two words are often used interchangeably in everyday language, but in data science they mean very different things. Confusing them is one of the most common conceptual errors students make in introductory statistics, and the confusion compounds later when you start training machine learning models.

### 1.1 Probability

**Probability** asks: *"Given a known model, how likely is this outcome?"*

You know the rules. You are asking about the data.

> **Example.** A coin is fair. What is the probability of getting 3 heads in a row?
> The coin's fairness (the model) is fixed and known. You are asking about the outcome.

Probability is always between 0 and 1:

- 0 means the event will never happen
- 1 means the event will always happen
- 0.5 means the event is equally likely to happen or not

### 1.2 Likelihood

**Likelihood** asks: *"Given the data I observed, how plausible is this model?"*

You know the outcome. You are asking about the model.

> **Example.** You flip a coin 10 times and get 9 heads. How plausible is it that this is a fair coin?
> The outcome is fixed (9 heads). You are evaluating the model (fair versus biased coin).

### 1.3 The Simple Way to Remember It

| | Fixed | Unknown |
|---|---|---|
| **Probability** | The model (parameters) | The outcome (data) |
| **Likelihood** | The outcome (data) | The model (parameters) |

This distinction becomes critical in machine learning. Model training is fundamentally a likelihood problem: you hold your observed data fixed and search for the model parameters that make that data most plausible. When someone says "maximum likelihood estimation," that is literally the procedure of finding the parameter values that maximise this likelihood.

## 2. Marginal Probability

### 2.1 What It Is

Marginal probability is the probability of a **single event occurring**, regardless of any other events. It is the simplest form of probability.

$$P(A)$$

The word "marginal" comes from the practice of summing across the margins of a probability table, exactly as you will do below.

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

These are called "marginal" because each one considers a single variable at a time, ignoring the other entirely. You are reading straight off the row or column totals.

### 2.3 An Important Sanity Check

For any single variable, the marginal probabilities of all its possible outcomes must sum to 1. P(Passed) + P(Failed) = 0.65 + 0.35 = 1.0. If they do not sum to 1, you have a computation error somewhere.

## 3. Joint Probability

### 3.1 What It Is

Joint probability is the probability of **two events occurring at the same time**.

$$P(A \cap B) = P(A \text{ and } B)$$

### 3.2 Example

Using the same table above:

- P(Passed **and** Attended) = 90 / 200 = **0.45**
- P(Failed **and** Did not attend) = 40 / 200 = **0.20**

The four joint probabilities in this table should sum to 1, because every student must fall into exactly one of the four cells: 0.45 + 0.15 + 0.20 + 0.20 = 1.00. (P(Passed and Did not attend) = 40/200 = 0.20; P(Failed and Attended) = 30/200 = 0.15.)

### 3.3 For Independent Events

If two events are **independent** (the occurrence of one has no bearing on the other), joint probability simplifies to a product:

$$P(A \cap B) = P(A) \times P(B)$$

> **Example.** The probability of rolling a 6 on a die *and* flipping heads on a coin:
> P(6) × P(Heads) = (1/6) × (1/2) = **1/12 ≈ 0.083**

> **Important.** This shortcut only works if the events are truly independent. Exam attendance and passing are *not* independent. Attending sessions clearly relates to passing, which is exactly why we need conditional probability to handle cases like this properly. Treating dependent events as if they were independent is one of the most common errors in applied probability.

### 3.4 How to Tell if Events Are Independent

Two events A and B are independent if and only if $P(A \cap B) = P(A) \times P(B)$. You can check this directly from your data. For the student table:

- $P(\text{Passed} \cap \text{Attended}) = 0.45$ (observed)
- $P(\text{Passed}) \times P(\text{Attended}) = 0.65 \times 0.60 = 0.39$ (predicted under independence)

These do not match (0.45 versus 0.39), so passing and attending are not independent. The fact that the observed joint probability is *higher* than the independence prediction tells us that attending and passing tend to occur together more often than chance alone would suggest.

## 4. Conditional Probability

### 4.1 What It Is

Conditional probability is the probability of event A occurring **given that event B has already occurred**.

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Read as: *"The probability of A, given B."*

### 4.2 Example

What is the probability that a student **passed**, given that they **attended sessions**?

$$P(\text{Passed} \mid \text{Attended}) = \frac{P(\text{Passed and Attended})}{P(\text{Attended})} = \frac{90/200}{120/200} = \frac{90}{120} = 0.75$$

Compare this to the marginal probability of passing (0.65, from Section 2). Attending sessions raises the probability of passing from 65% to 75%, a meaningful difference, and clear evidence that the two events are *not* independent (which is exactly why the shortcut in Section 3.3 does not apply here).

Conditional probability is how we formally model **context and new information**. It is the mathematical tool behind statements like "given what we already know, how should we update our expectations?"

### 4.3 A Subtlety Worth Internalising

$P(A \mid B)$ and $P(B \mid A)$ are not the same thing. They use the same two events but ask very different questions. From our student data:

- $P(\text{Passed} \mid \text{Attended}) = 90 / 120 = 0.75$
- $P(\text{Attended} \mid \text{Passed}) = 90 / 130 = 0.69$

Both are valid conditional probabilities, but they answer different questions. The first asks "of students who attended, what fraction passed?" The second asks "of students who passed, what fraction attended?" Confusing one for the other is a very common mistake (sometimes called the "prosecutor's fallacy" in legal contexts) and the next section, Bayes' theorem, is exactly the formal tool that converts between them.

## 5. Bayes' Theorem

### 5.1 What It Does

Bayes' theorem flips a conditional probability around. It lets you update your beliefs about something unobservable once new evidence arrives. It is derived directly from the conditional probability formula in Section 4:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

In plain English:

> *"How likely is A given B?"* equals *"How likely is B given A?"* times *"How likely was A in the first place?"* divided by *"How likely is B overall?"*

### 5.2 The Three Components

| Term | Name | Meaning |
|---|---|---|
| P(A) | **Prior** | Your belief about A before seeing any evidence |
| P(B\|A) | **Likelihood** | How probable the evidence is, assuming A is true |
| P(A\|B) | **Posterior** | Your updated belief about A after seeing the evidence |

Notice that "likelihood" here is exactly the concept introduced in Section 1.2. Bayes' theorem is, in effect, the formal bridge between probability and likelihood: it tells you precisely how to convert a likelihood into an updated probability, as long as you also account for your prior belief.

### 5.3 The Law of Total Probability

The denominator $P(B)$ often needs to be expanded using the **law of total probability**. If A and "not A" (written $\neg A$) together cover all possibilities:

$$P(B) = P(B \mid A) \cdot P(A) + P(B \mid \neg A) \cdot P(\neg A)$$

This is the version of $P(B)$ you will use most often when applying Bayes' theorem in practice, especially in the medical-testing-style problems where the evidence (a positive test) can occur for two distinct reasons (you actually have the disease, or you got a false positive).

### 5.4 Intuitive Example: Medical Testing

A disease affects 1% of the population. A test for it is 95% accurate (true positive rate), and it has a 5% false positive rate. If someone tests positive, what is the probability they actually have the disease?

Most people's gut instinct is "about 95%." The real answer is much lower. Working through why is one of the most valuable intuition-builders in all of applied probability.

The three quantities Bayes' theorem needs:

- $P(\text{Disease}) = 0.01$ (the prior, 1% of the population has it)
- $P(\text{Positive} \mid \text{Disease}) = 0.95$ (the likelihood, the test is 95% sensitive)
- $P(\text{Positive} \mid \neg\text{Disease}) = 0.05$ (the false positive rate)

Applying the law of total probability:

$$P(\text{Positive}) = (0.95)(0.01) + (0.05)(0.99) = 0.0095 + 0.0495 = 0.059$$

Now Bayes':

$$P(\text{Disease} \mid \text{Positive}) = \frac{0.95 \times 0.01}{0.059} \approx 0.161$$

**Only about a 16% chance of actually having the disease**, even with a positive test. Because the disease is rare to begin with, the much larger pool of healthy people who test falsely positive ends up outnumbering the smaller pool of genuinely sick people who test correctly positive. This is Bayes' theorem in action: it forces you to weigh the **prior probability**, not just the accuracy of the evidence in isolation.

> **A counter-intuitive lesson.** This is also why population-level screening programs are evaluated so carefully. A highly accurate test can still produce mostly false alarms when the condition it detects is rare. Following up an apparent positive with a more specific (often more expensive) confirmatory test is therefore standard practice in medicine.

## 6. Connecting Distributions Back to Probability

Yesterday's distributions, Uniform, Poisson, and Normal, are essentially **probability models**. They describe exactly how probability is spread across all possible outcomes of a variable. Everything you learned about marginal, joint, and conditional probability above can be applied directly to them.

### 6.1 Discrete Distributions and Probability Mass Functions (PMF)

For discrete outcomes (counts, categories), the distribution gives you the exact probability of each value. This is precisely the PMF introduced in yesterday's Section 2.1.

> **Poisson example.** If calls arrive at λ = 4 per hour, the Poisson distribution can answer:
>
> - P(exactly 3 calls) = ?
> - P(more than 6 calls) = ?

### 6.2 Continuous Distributions and Probability Density Functions (PDF)

For continuous outcomes (heights, weights, scores), you cannot meaningfully ask for the probability of one exact value. As covered yesterday, that probability is effectively zero. Instead, you ask about **ranges**, using the PDF.

> **Normal example.** If exam scores are N(70, 10), what is the probability a student scores between 60 and 80?
> Answer: about 68%. The 68-95-99.7 rule from yesterday's lesson is itself just a probability statement about a range.

### 6.3 The Key Connection

Once you know what distribution your data follows, you can make precise probability statements about future observations, anomalies, and expected ranges. You can also combine that with everything in this lesson (joint, conditional, Bayes') to reason about more complex, multi-variable questions. This is why distributions and probability theory are always taught together: distributions are the *machinery* of probability, and probability is the *interpretation* of distributions.

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

```python
# Independence check
expected_under_indep = p_passed * p_attended
print(f"P(Passed) x P(Attended) under independence: {expected_under_indep:.4f}")
print(f"Actual P(Passed AND Attended):              {p_passed_and_attended:.4f}")
print("The two values differ, so passing and attending are not independent.")
```

### 7.3 Conditional Probability

```python
# P(Passed | Attended)
p_passed_given_attended = df.loc['Attended', 'Passed'] / df.loc['Attended'].sum()
print(f"P(Passed | Attended): {p_passed_given_attended:.2f}")

# P(Passed | Did not attend)
p_passed_given_not = df.loc['Did_not_attend', 'Passed'] / df.loc['Did_not_attend'].sum()
print(f"P(Passed | Did not attend): {p_passed_given_not:.2f}")
```

```python
# Visualise the difference
labels = ['Attended', 'Did not attend']
probs = [p_passed_given_attended, p_passed_given_not]

plt.figure(figsize=(7, 4))
sns.barplot(x=labels, y=probs, palette="muted")
plt.axhline(p_passed, color='red', linestyle='--', label=f'Overall P(Passed) = {p_passed:.2f}')
plt.title("P(Passed | Attendance Status)")
plt.ylabel("Probability of Passing")
plt.ylim(0, 1)
plt.legend()
plt.show()
```

```python
# Note that P(A|B) is not the same as P(B|A)
p_attended_given_passed = df.loc['Attended', 'Passed'] / df['Passed'].sum()
print(f"P(Passed | Attended): {p_passed_given_attended:.2f}")
print(f"P(Attended | Passed): {p_attended_given_passed:.2f}")
print("These are different conditional probabilities and answer different questions.")
```

### 7.4 Bayes' Theorem: The Medical Test Example

```python
# Known probabilities
p_disease = 0.01               # Prior: 1% prevalence
p_pos_given_disease = 0.95     # True positive rate (sensitivity)
p_pos_given_no_disease = 0.05  # False positive rate

# Law of total probability for P(Positive)
p_no_disease = 1 - p_disease
p_positive = (p_pos_given_disease * p_disease) + (p_pos_given_no_disease * p_no_disease)

# Bayes: P(Disease | Positive)
p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

print(f"P(Positive): {p_positive:.4f}")
print(f"P(Disease | Positive test): {p_disease_given_positive:.4f} ({p_disease_given_positive*100:.1f}%)")
```

> Try changing `p_disease` to 0.10 (10% prevalence) and observe how dramatically the result changes. This is why screening programs depend so heavily on the prevalence of the condition being tested for, not just the test's accuracy.

```python
# Sensitivity sweep: how does the posterior depend on the prior?
priors = np.linspace(0.001, 0.5, 100)
posteriors = []
for p in priors:
    denom = p_pos_given_disease * p + p_pos_given_no_disease * (1 - p)
    posteriors.append((p_pos_given_disease * p) / denom)

plt.figure(figsize=(9, 4))
plt.plot(priors, posteriors, color='steelblue', lw=2)
plt.axhline(0.5, color='gray', linestyle=':', label='Posterior = 50%')
plt.xlabel("Prior probability of disease, P(Disease)")
plt.ylabel("Posterior P(Disease | Positive)")
plt.title("How Prior Prevalence Affects the Meaning of a Positive Test")
plt.legend()
plt.show()
```

The posterior probability of disease climbs steeply with the prior. This is the Bayesian logic that medical professionals apply when ordering tests in a high-risk population: the same test result means very different things depending on the patient's prior risk profile.

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

# Probability of scoring below 50
p_below_50 = dist.cdf(50)
print(f"P(score < 50): {p_below_50:.4f} ({p_below_50*100:.1f}%)")
```

```python
# Visualise
x = np.linspace(40, 100, 300)
y = dist.pdf(x)

plt.figure(figsize=(9, 4))
plt.plot(x, y, color='steelblue', lw=2)
plt.fill_between(x, y, where=(x >= 60) & (x <= 80), alpha=0.4, color='steelblue', label='P(60 to 80)')
plt.fill_between(x, y, where=(x >= 90), alpha=0.4, color='tomato', label='P(>90)')
plt.fill_between(x, y, where=(x <= 50), alpha=0.4, color='orange', label='P(<50)')
plt.axvline(mu, color='black', linestyle='--', label='Mean')
plt.title("Normal Distribution: Exam Scores")
plt.xlabel("Score")
plt.ylabel("Density")
plt.legend()
plt.show()
```

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

# Probability of zero tickets (a totally quiet hour)
p_zero = dist_poisson.pmf(0)
print(f"P(0 tickets): {p_zero:.4f}")
```

```python
# Plot the PMF
k = np.arange(0, 15)
pmf = dist_poisson.pmf(k)

plt.figure(figsize=(8, 4))
plt.bar(k, pmf, color='darkorchid', alpha=0.8)
plt.axvline(lam, color='red', linestyle='--', label=f'λ = {lam}')
plt.title("Poisson Distribution: Support Tickets per Hour")
plt.xlabel("Number of Tickets")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

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

# Mean wait time
print(f"Expected wait time: {dist_uniform.mean():.1f} minutes")
```

Notice that unlike the Normal and Poisson examples, every equally-sized interval here has the *same* probability. This is a direct consequence of yesterday's "flat histogram" shape.

## Daily Challenge

1. A company's data shows that 40% of customers churn. Among churned customers, 70% had contacted support more than 3 times. Among retained customers, only 20% had done so. A new customer has contacted support 4 times. Use Bayes' theorem to calculate the probability they will churn. Compare this posterior to the prior (the 40% baseline churn rate) and interpret the change.

2. Using `scipy.stats.norm`, find the probability that a randomly selected student scores above 85 in an exam with mean = 70 and std = 10. Shade this region on a plot. Then find the score threshold below which only the bottom 10% of students fall (use `ppf`, the inverse CDF).

3. A factory produces defects at an average rate of 2 per batch (λ = 2). Using the Poisson distribution, calculate:
   - The probability of zero defects in a batch
   - The probability of more than 4 defects
   - The smallest defect count `k` for which P(defects ≥ k) is below 0.05 (this would be a useful threshold for triggering a quality alert)

4. Build a 2×2 contingency table from a dataset of your choice (or invent a plausible scenario). Compute all four marginal probabilities, all four joint probabilities, and both relevant conditional probabilities. State in one sentence whether the two variables appear independent, with evidence.

5. **Bonus.** Revisit the medical testing example in Section 5.4. Suppose a more accurate test exists with sensitivity 99% and a false positive rate of 1%. At the 1% prevalence rate, what is the posterior probability of disease given a positive test now? How much improvement does this represent over the original 95% / 5% test?

> Submit your notebook to the Kanban board under your name by end of day.

## Summary

| Concept | Question it answers | Key formula |
|---|---|---|
| **Probability** | How likely is this outcome given a known model? | P(A) |
| **Likelihood** | How plausible is this model given the data? | L(model \| data) |
| **Marginal Probability** | How likely is one event on its own? | P(A) |
| **Joint Probability** | How likely are two events together? | P(A ∩ B) |
| **Conditional Probability** | How likely is A, given B already happened? | P(A\|B) = P(A∩B) / P(B) |
| **Independence test** | Do two events have no effect on each other? | P(A∩B) = P(A) × P(B) |
| **Bayes' Theorem** | How do I update my belief with new evidence? | P(A\|B) = P(B\|A)·P(A) / P(B) |
| **Law of Total Probability** | How do I expand P(B) when needed for Bayes'? | P(B) = P(B\|A)P(A) + P(B\|¬A)P(¬A) |

## What's Next

Tomorrow's lesson moves into **Hypothesis Testing**, where we will use everything you have learned this week (distributions, probability, and the sampling distribution from Monday) to make formal decisions from data. You will learn how to think about Type I and Type II errors, statistical significance, and how to determine whether a difference you observe is real or just due to chance.

*Lesson designed for internal use, Statistical Foundations, Data Science Curriculum*
