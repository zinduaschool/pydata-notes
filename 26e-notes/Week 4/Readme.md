# Statistics & Math for Data Science — Week Overview

> **Course:** Data Science Foundations
> **Module:** Statistics & Math for Data Science (1-Week Intensive)
> **Prerequisites:** Basic Python, NumPy/Pandas familiarity

---

## Purpose of This Module

This week takes you from describing a dataset to reasoning about it under uncertainty, and finally to the mathematical machinery that powers the models you'll build later in the course. Each day builds directly on the one before it — by Saturday, you'll be able to trace a straight line from "what does my data look like?" all the way to "how does a model actually learn from it?"

---

## Week Structure

| Day | File | Topic | Core Concepts |
|---|---|---|---|
| **Monday** | `1-statistics-fundamentals.md` | Statistics Fundamentals | Central tendency, spread, relationships (covariance & correlation), descriptive vs. inferential statistics |
| **Tuesday** | `2-distributions.md` | Distributions | Discrete vs. continuous distributions, Uniform, Poisson, Normal — shapes and identification |
| **Wednesday** | `3-probability-theory.md` | Probability Theory | Probability vs. likelihood, joint probability, conditional probability, Bayes' theorem |
| **Thursday** | `4-hypothesis-testing.md` | Hypothesis Testing | Type I/II errors, significance, p-values, statistical decision-making |
| **Friday** | `5-linear-algebra-essentials.md` | Math for Data Science I | Vectors, matrices, matrix multiplication, normal equations |
| **Saturday** | `6-calculus-essentials.md` | Math for Data Science II | Derivatives, partial derivatives, gradients, optimization intuition |

---

## File Naming Convention

Files are numbered in the order they should be taught, using the pattern `N-topic-name.md`. This mirrors the structure of the existing curriculum materials, so this module slots in as a direct continuation — `1-statistics-fundamentals.md` picks up where prior foundational material leaves off and feeds directly into `2-distributions.md`, and so on through the week.

---

## How the Week Connects

```
Mon: Stats Fundamentals  →  Tue: Distributions  →  Wed: Probability Theory
        │                         │                         │
   Describe data            Model the data            Quantify uncertainty
   (mean, spread,           (Uniform, Poisson,         (joint, conditional,
   relationships)            Normal)                    Bayes' theorem)
                                                               │
                                                               ▼
                                                  Thu: Hypothesis Testing
                                                  Make decisions from data
                                                  (errors, significance, p-values)
                                                               │
                                                               ▼
                                          Fri/Sat: Math for Data Science
                                          The engine under the hood
                                          (linear algebra → calculus → optimization)
```

Each day's lesson follows the same structure: **Learning Objectives → Concept Walkthroughs → Hands-On Python → Daily Challenge → Summary → What's Next**.
---

## Daily Workflow

- **Concept sessions** : Theory, worked examples, and intuition-building — each new idea is tied back to something covered earlier in the week.
- **Hands-on sessions** : Python implementation using NumPy, Pandas, SciPy, Matplotlib, and Seaborn on real and simulated datasets.
- **Daily Challenge**: A short applied exercise submitted to the Kanban board by end of day, designed to reinforce that day's concepts before the next day builds on them.
- **Lab**: A guided notebook in `labs/` with `# TODO` cells, self-check cells, and written questions. Each lab uses a different dataset from the lesson, so learners adapt the ideas rather than copy code.

## Labs

| Day | Lab notebook | Dataset | Extends the lesson with |
|---|---|---|---|
| Mon | `labs/1-statistics-fundamentals-lab.ipynb` | `mpg` | Stats from scratch, missing-value handling, Spearman correlation, standard error $\sigma/\sqrt{n}$ |
| Tue | `labs/2-distributions-lab.ipynb` | `taxis` | Mystery distributions, Bernoulli/Binomial, z-scores, QQ plots, overdispersion in real counts |
| Wed | `labs/3-probability-theory-lab.ipynb` | `titanic` + dice | Complement/addition rules, Bayes on real data, sequential updating, Monte Carlo checks, likelihood/MLE |
| Thu | `labs/4-hypothesis-testing-lab.ipynb` | `penguins` | One-sample t-test by hand, confidence intervals, Welch's test, Cohen's d, ANOVA, permutation test, multiple testing, sample-size planning |
| Fri | `labs/5-linear-algebra-lab.ipynb` | `mpg` | Loop-based matrix multiplication, inverse and singular matrices, cosine similarity, split-then-scale, collinearity |
| Sat | `labs/6-calculus-lab.ipynb` | `mpg` | Finite-difference error, exp/log/sigmoid derivatives, gradient checking, learning-rate limits, local minima, mini-batch SGD |

Worked solutions and model answers are in `labs/solutions/`. Attempt each lab on your own first, then use the solutions to check your work or to get unstuck. Labs need `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, and `scikit-learn` (Fri/Sat), plus internet access the first time `sns.load_dataset` runs.

--
