# Undergraduate Technical Archive — Bocconi University

This repository archives research, mathematical proofs, and applied machine learning implementations developed during my Bachelor’s degree at Bocconi University. The work spans stochastic processes, quantum machine learning, geometric deep learning, and quantitative methods, split between independent research, coursework, and student organizations (**BAINSA** and **BSDSA**).

---

## Core Research & Algorithmic Highlights

* **Stochastic Processes & Group Theory:** Formal proof and $\mathcal{O}(1)$ CRT-vectorized simulation for the linear expected hitting time $\mathbb{E}[C_{x^*}] = \mathcal{O}(\vert{}S\vert{})$ of a blind ballistic agent on an unknown discrete torus $\mathbb{Z}_N \times \mathbb{Z}_M$.
* **Quantum Machine Learning & Evolutionary QCA:** Evolutionary circuit synthesis and gradient-free optimization of parameterized quantum circuits (PQCs) using `PennyLane` and `JAX` (`vmap`/`jit`).
* **Geometric Deep Learning:** Implementation of Graph Transformer Convolutional Networks and custom layers ([arXiv:1610.04325](https://arxiv.org/abs/1610.04325)) for small-molecule protein binding affinity prediction (Kaggle BELKA).
* **Non-Convex Stochastic Optimization:** Softmax-biased simulated annealing framework for 2D grid optimization with topological landscape analysis.

---

## Repository Directory

| Directory | Domain | Key Methods & Stack |
| :--- | :--- | :--- |
| [`Markov Torus Hitting Time/`](./Markov%20Torus%20Hitting%20Time) | Markov Chains, Spectral Graph Theory | Fourier analysis on abelian groups, spectral gap bounds, CRT vectorization |
| [`BELKA Competition/`](./BELKA%20Competition) | Geometric Deep Learning, Cheminformatics | Graph Transformers, custom GNN message passing, parallel data loading |
| [`Course Related Projects/Computational Modelling/`](./Course%20Related%20Projects/Computational%20Modelling) | Quantum Machine Learning | Evolutionary QCA, gradient-free PQC training, `JAX`, `PennyLane` |
| [`Course Related Projects/Computer Programming/`](./Course%20Related%20Projects/Computer%20Programming) | Stochastic Optimization | Softmax-biased Simulated Annealing, landscape topology |
| [`BAINSA/`](./BAINSA) | Computer Vision, Signal Processing | Recursive SSIM/MSE degradation analysis, YOLOv8 object detection |
| [`BSDSA/`](./BSDSA) | Applied ML & Vision | CNN feature map extraction, circuit recognition, hyperparameter tuning |
| [`BERT_Project/`](./BERT_Project) | Natural Language Processing | Fine-tuned BERT for financial headline sentiment classification |
| [`Course Related Projects/Machine Learning/`](./Course%20Related%20Projects/Machine%20Learning) | Tabular ML, Ensemble Methods | Randomer Forests, genetic feature selection, data imputation |
| [`Course Related Projects/Econometrics/`](./Course%20Related%20Projects/Econometrics) & [`Finance/`](./Course%20Related%20Projects/Finance) | Quantitative Finance & Econometrics | Statistical modeling, hypothesis testing, financial time series (`R`, `Python`) |

---

### 1. Blind Ballistic Search on the Discrete Torus
**Location:** [`Markov Torus Hitting Time/`](./Markov%20Torus%20Hitting%20Time)

Formal proof and numerical validation for the expected hitting time of a blind agent searching an unknown 2D discrete torus $S = \mathbb{Z}_N \times \mathbb{Z}_M$ under a bounded aspect ratio $M \le R \cdot N$. Because the state-space dimensions and initial coordinates are hidden from the agent, direct uniform sampling is impossible, while standard diffusive random walks on 2D tori suffer from a $\Theta(\vert{}S\vert{} \log \vert{}S\vert{})$ cover/hitting overhead.

* **Multi-Scale Ballistic Algorithm:** The search proceeds in dyadic phases $j \ge 1$, with the agent positing a maximum torus dimension $D_j = 2^j$. Within phase $j$, the agent executes independent ballistic epochs by drawing a random axial or diagonal generator $v \in \{(1,0), (0,1), (1,1), (1,-1)\}$ and a uniform trajectory length $L \sim \text{Unif}\ \\{1, \dots, D_j\\}$, inspecting every vertex along the resulting ray. If the target $x^* \in S$ is not found within a phase budget of $\Theta(D_j^2)$ total steps, the agent increments $j \leftarrow j + 1$ and quadruples the step allocation.
* **Fourier-Analytic Proof & Linear Hitting Bound:** Treating the sequence of epoch starting states as a discrete-time Markov chain on the finite abelian group $S$, the transition operator acts as a group convolution diagonalized by the characters $\chi_{a,b}(x, y) = \omega_N^{ax} \omega_M^{by}$. This structure yields an optimal linear expected hitting cost:
  $$\mathbb{E}[C_{x^*}] = \mathcal{O}(\vert{}S\vert{})$$
  * *Uniform Spectral Gap:* Once the scale reaches $D_j \ge D_{j^*} \approx \max(N, M)$, the non-trivial Fourier eigenvalues are bounded by $\lambda_* \le 7/8$ uniformly across all subsequent phases $j \ge j^*$, establishing an $\mathcal{O}(1)$ epoch mixing time independent of $\vert{}S\vert{}$.
  * *Second-Moment Block Hitting:* Rather than relying on stationary hitting time bounds that break down for non-reversible or normal chains, a second-moment Paley–Zygmund argument over epoch blocks bounds the single-block miss probability strictly away from $1$ from any arbitrary starting state.
  * *Doubly Exponential Tail Decay:* Because the number of epochs per phase scales as $\Theta(D_j)$, the probability of failing to locate $x^*$ in phase $j = j^* + k$ decays doubly exponentially as $\exp(-\beta 2^k)$. This rapid tail suppression dominates the $4^k$ geometric growth in phase step budgets, forcing the expected cost series to converge to $\mathcal{O}(D_{j^*}^2) = \mathcal{O}(\vert{}S\vert{})$.
* **$\mathcal{O}(1)$ CRT-Vectorized Simulation:** To avoid the $\mathcal{O}(L)$ bottleneck of step-by-step trajectory simulation on large grids, ray-target intersections are computed analytically in $\mathcal{O}(1)$ time per epoch. Diagonal traversals reduce to solving the simultaneous modular congruences $l \equiv \Delta x \pmod N$ and $l \equiv \Delta y \pmod M$ for the minimal positive step $l \le L$ via the generalized Chinese Remainder Theorem. This reduction enables numerical verification up to $\vert{}S\vert{} \approx 10^9$ across aspect ratios $R \in [1, 10]$, where the empirical hitting cost stabilizes tightly between $1.15\vert{}S\vert{}$ and $1.30\vert{}S\vert{}$.

---

### 2. Molecular Binding Affinity Prediction (BELKA Competition)
**Location:** [`BELKA Competition/`](./BELKA%20Competition)

Deep learning pipeline designed for the Kaggle BELKA competition to predict small-molecule binding affinity across protein targets.
* **Architecture:** Implemented Graph Transformer Convolutional Networks alongside custom GNN layers based on [arXiv:1610.04325](https://arxiv.org/abs/1610.04325).
* **Systems Engineering:** Built custom multi-threaded data loaders and parallelized preprocessing pipelines to handle massive molecular graph datasets under memory constraints.

---

### 3. Student Association Research (BAINSA & BSDSA)
**Locations:** [`BAINSA/`](./BAINSA) | [`BSDSA/`](./BSDSA)

Applied research and engineering sprints conducted within Bocconi's AI, Neuroscience, and Data Science associations:
* **Recursive Image Degradation (`BAINSA/SSIMPaper`):** Analyzed structural similarity (SSIM) and MSE decay under recursive downscaling/upscaling and generative enhancement (`Remini`) using `JAX`-accelerated array operations.
* **Real-Time Object Detection (`BAINSA/YOLOPaper`):** Architecture evaluation and deployment pipelines for YOLO-based detection.
* **Computer Vision & AutoML (`BSDSA/`):** Hand-drawn electrical circuit component recognition, intermediate CNN feature map extraction/interpretability, and automated hyperparameter optimization.

---

### 4. Coursework & Independent Implementations
**Locations:** [`Course Related Projects/`](./Course%20Related%20Projects) | [`BERT_Project/`](./BERT_Project)

* **Computational Modelling (`Computational Modelling/`):** Evolutionary design of Quantum Cellular Automata (QCA) and gradient-free optimization of stochastic quantum circuits using `PennyLane` and `JAX`. Includes the full paper (`Evolutionary Design of QCA.pdf`) and circuit depth/loss convergence benchmarks.
* **Stochastic Optimization (`Computer Programming/`):** Custom Softmax-Biased Simulated Annealing (`SimAnnFinal.py`) for multi-modal grid landscapes, paired with empirical convergence analysis (`Project Report.pdf`).
* **Machine Learning (`Machine Learning/`):** Credit risk classification (`Credit Classification.ipynb`) utilizing Randomer Forests, evolutionary feature engineering, deprecated library restoration, and non-trivial missing-data imputation.
* **Financial NLP (`BERT_Project/`):** Fine-tuned BERT architectures for sentiment classification on financial news headlines to extract market-moving sentiment signals.
* **Econometrics & Quantitative Finance (`Econometrics/`, `Finance/`):** Empirical econometric modeling in `R` and quantitative portfolio/financial analysis in Python.
