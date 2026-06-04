# Hidden Markov Models — Viterbi, MLE & Bioinformatics

> M1 Informatique — Parcours Intelligence Artificielle · Université d'Avignon · Mars 2025

[![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus)](https://isocpp.org)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![HMM](https://img.shields.io/badge/Model-Hidden%20Markov%20Model-purple)]()

---

## Overview

This project implements **Hidden Markov Models (HMM)** from scratch in C++, covering three core algorithms:
1. **Viterbi** — optimal state sequence decoding
2. **MLE (Maximum Likelihood Estimation)** — supervised parameter learning
3. **Baum-Welch** — unsupervised parameter estimation (EM algorithm)

Applications include meteorological sequence prediction and **CG-island detection** in genomic sequences.

---

## Table of Contents

- [Background](#background)
- [Project Structure](#project-structure)
- [Algorithms](#algorithms)
  - [1. Viterbi Algorithm](#1-viterbi-algorithm)
  - [2. MLE Training](#2-mle-training)
- [Experiments & Results](#experiments--results)
  - [Weather Sequence Prediction](#weather-sequence-prediction)
  - [CG-Island Detection in Genomics](#cg-island-detection-in-genomics)
- [Key Results](#key-results)
- [Build & Usage](#build--usage)
- [Author](#author)

---

## Background

A **Hidden Markov Model** $\lambda = (\pi, A, B)$ is defined by:
- $\pi_i = P(q_1 = s_i)$ — initial state distribution
- $A_{ij} = P(q_{t+1} = s_j \mid q_t = s_i)$ — transition matrix
- $B_{jk} = P(o_t = v_k \mid q_t = s_j)$ — emission matrix

All computations are done in **log-space** to avoid numerical underflow on long sequences.

---

## Project Structure

```
hidden-markov-models-cpp/
├── src/
│   ├── model.cpp          # HMM core (Viterbi, MLE, Baum-Welch)
│   ├── model.h
│   ├── use_hmm.cpp        # CLI runner (train/test/label)
│   └── compErrors.py      # Error rate evaluation script
├── data/
│   ├── weather/           # Meteorological dataset
│   │   ├── weather.true.model
│   │   ├── weather-test.csv
│   │   └── weather-test.ref
│   └── genetics/          # Genomic CG-island dataset
│       ├── CG-region.model
│       ├── CG-region.csv
│       └── CG-region.ref
├── rapport_TP2_HMM.tex    # Full technical report (LaTeX)
├── MLE-HMM.pdf            # Reference paper
├── hmm.pdf                # Course notes
└── README.md
```

---

## Algorithms

### 1. Viterbi Algorithm

Finds the most probable hidden state sequence $Q^* = \arg\max_Q P(Q, O \mid \lambda)$ using dynamic programming in log-space:

**Initialization** ($t = 0$):
$$\delta_0(i) = \log \pi_i + \log B_{i,o_0}, \quad \psi_0(i) = 0$$

**Recursion** ($t = 1, \ldots, T-1$):
$$\delta_t(j) = \max_i \left[\delta_{t-1}(i) + \log A_{ij}\right] + \log B_{j,o_t}$$
$$\psi_t(j) = \arg\max_i \left[\delta_{t-1}(i) + \log A_{ij}\right]$$

**Backtracking**: $q^*_t = \psi_{t+1}(q^*_{t+1})$ from $t = T-2$ down to $0$.

```cpp
// Core recursion (C++)
for (int t = 1; t < T; t++) {
    for (int j = 0; j < n; j++) {
        double best = -INFINITY;
        int best_i = 0;
        for (int i = 0; i < n; i++) {
            double val = delta[t-1][i] + A[i][j];
            if (val > best) { best = val; best_i = i; }
        }
        delta[t][j] = best + B[j][seq_obs[t]];
        psi[t][j]   = best_i;
    }
}
```

### 2. MLE Training

Given $N$ labeled sequences $\{(O^{(k)}, Q^{(k)})\}_{k=1}^N$, the estimators are:

$$\hat{\pi}_i = \frac{\sum_k \mathbf{1}[q_1^{(k)} = s_i]}{N}$$

$$\hat{a}_{ij} = \frac{\sum_k \sum_t \mathbf{1}[q_t^{(k)}=s_i,\, q_{t+1}^{(k)}=s_j]}{\sum_k \sum_t \mathbf{1}[q_t^{(k)}=s_i]}$$

$$\hat{b}_{jk} = \frac{\sum_k \sum_t \mathbf{1}[q_t^{(k)}=s_j,\, o_t^{(k)}=v_k]}{\sum_k \sum_t \mathbf{1}[q_t^{(k)}=s_j]}$$

All stored as log-probabilities to avoid underflow.

---

## Experiments & Results

### Weather Sequence Prediction

**Model:** 3 hidden states (`chaud`, `doux`, `froid`), learned from labeled meteorological sequences.

```bash
./use_hmm test ../data/weather/weather.true.model \
               ../data/weather/weather-test.csv \
               /tmp/weather-pred.txt
python3 compErrors.py ../data/weather/weather-test.ref /tmp/weather-pred.txt
```

**Result:**

| Metric | Value |
|--------|-------|
| Test sequences | 18 observations |
| Errors | **0 / 18** |
| Error rate | **0.00%** |
| Accuracy | **100%** |

The Viterbi decoder perfectly recovers the hidden weather states on the test set.

---

### CG-Island Detection in Genomics

**Task:** Segment a DNA sequence of 100,000 bases into high-GC and low-GC regions.

**HMM parameters:**

| Transition | Probability |
|------------|-------------|
| High GC → High GC | 0.99 |
| High GC → Low GC  | 0.01 |
| Low GC → High GC  | 0.01 |
| Low GC → Low GC   | 0.99 |

| State | P(A) | P(G) | P(C) | P(T) |
|-------|------|------|------|------|
| High GC (+) | 0.20 | 0.30 | 0.30 | 0.20 |
| Low GC (−)  | 0.30 | 0.20 | 0.20 | 0.30 |

**Result:**

| Metric | Value |
|--------|-------|
| Test bases | 100,000 |
| Errors | 16,685 / 100,000 |
| Error rate | **16.68%** |
| Accuracy | **83.32%** |

The 16.68% error rate is symmetric across both states (~16.7% each), reflecting the difficulty of distinguishing states with close emission distributions ($\Delta P \approx 0.10$ per nucleotide).

---

## Key Results

| Task | Metric | Value |
|------|--------|-------|
| Weather prediction | Accuracy | **100%** (0 errors / 18) |
| CG-island detection | Accuracy | **83.32%** (100K bases) |
| CG-island detection | Error rate | 16.68% (symmetric ±16.7% per state) |

---

## Build & Usage

```bash
# Compile
cd src
g++ -O2 -std=c++17 -o use_hmm use_hmm.cpp model.cpp

# Test Viterbi decoding
./use_hmm test ../data/weather/weather.true.model \
               ../data/weather/weather-test.csv \
               /tmp/pred.txt

# Evaluate
python3 compErrors.py ../data/weather/weather-test.ref /tmp/pred.txt

# Train MLE from labeled data
./use_hmm train ../data/train-obs.csv ../data/train-states.csv output.model
```

---

## Author

**Christ Kekeli KELI**  
M1 Informatique — Intelligence Artificielle  
Université d'Avignon · Mars 2025  
Encadrant : Yezekael HAYEL
