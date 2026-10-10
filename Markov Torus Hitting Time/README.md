# Blind Ballistic Search on the Discrete Torus

This repository contains the formal proof and numerical validation for the expected hitting time of a blind agent searching an unknown discrete torus. I originally encountered this problem in December 2024. After letting it sit for a while, I recently returned to develop the final formal solution and asymptotic bounds presented here.

## The Problem

An agent is placed on a 2-dimensional discrete torus $S=\mathbb{Z}_N\times\mathbb{Z}_M$. The dimensions $N$ and $M$ are completely hidden from the agent, though we assume a bounded aspect ratio $M \le R \cdot N$. The agent does not know its starting coordinates and must locate an arbitrary target $x^* \in S$.

Because the maximum dimension $D = M$ is unknown, a simple uniform sampling approach is impossible. Standard diffusive random walks on tori scale poorly, taking $\Theta(\vert{}S\vert{} \log \vert{}S\vert{})$ or worse. The proposed search algorithm instead operates via multi-scale ballistic "billiard" streaks:

* The algorithm runs in phases, with the agent guessing a maximum dimension $D_j = 2^j$.


* During a phase, the agent picks a random axial or diagonal generator and a random streak length uniformly distributed in $[1, D_j]$, checking every node along the continuous linear traversal.


* If the target is not found within a step budget of $\Theta(D_j^2)$, the agent increments $j$ and quadruples the budget for the next phase.



## The Formal Guarantee

The core theoretical result (`Snake_Game_Hitting_Time.pdf`) proves that this adaptive doubling schedule achieves a linear expected hitting time:


$$\mathbb{E}[C_{x^*}] = \mathcal{O}(\vert{}S\vert{})$$

The proof models the sequence of epoch starting points as a discrete-time Markov chain on the abelian group $S$. Because the transition kernel is a group convolution operator, it is diagonalized by the Fourier characters of $\mathbb{Z}_N \times \mathbb{Z}_M$. Key steps in the proof include:

* **Spectral Gap:** Bounding the non-trivial eigenvalues $\lambda_* \le 7/8$ uniformly across all phases $j \ge j^*$ to demonstrate a constant relaxation time.


* **Second-Moment Method:** Utilizing a worst-case block hitting lemma to bound the failure probability away from 1, bypassing the standard $\mathbb{E}_\pi[T_{x^*}]$ limitations for normal chains.


* **Doubly Exponential Decay:** Showing that the probability of failing a phase decays as $\exp(-\beta 2^k)$, which safely overpowers the geometric $4^k$ growth in step cost, forcing the series to converge to $\mathcal{O}(D_{j^*}^2) = \mathcal{O}(\vert{}S\vert{})$.



## Empirical Validation

The repository includes a Python simulation script to numerically verify the analytical bounds.

To bypass the bottleneck of linear point-by-point traversal, the simulator vectorizes the intersections using modular arithmetic. For diagonal rays, it computes the minimal positive solution $l$ to the congruences $l \equiv \Delta x \pmod N$ and $l \equiv \Delta y \pmod M$ in $\mathcal{O}(1)$ time via the Chinese Remainder Theorem.

This allows the simulation to scale effortlessly up to state spaces of size $\vert{}S\vert{} \approx 10^9$. The empirical results confirm that across five orders of magnitude and randomized aspect ratios $R \in [1, 10]$, the expected step cost stabilizes at a constant factor of the state space size (typically between $1.15\vert{}S\vert{}$ and $1.30\vert{}S\vert{}$).
