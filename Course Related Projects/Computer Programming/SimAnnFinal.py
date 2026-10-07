import numpy as np
import matplotlib.pyplot as plt


def accept_with_prob(delta_cost, beta):
    """Metropolis rule: True with probability min{1, exp(-beta * delta_cost)}."""
    if delta_cost <= 0:
        return True
    return np.random.random() < np.exp(-beta * delta_cost)


def two_segment_schedule(beta0, beta1, anneal_steps):
    """
    The report's finite schedule (Section 4): the first anneal_steps // 3 values
    go linearly from beta0 to log10(beta1), the remaining ones linearly from
    log10(beta1) to beta1.
    """
    if anneal_steps < 1:
        raise ValueError("anneal_steps must be at least 1")
    if beta1 <= 0:
        raise ValueError("beta1 must be positive")
    if not 0 <= beta0 <= np.log10(beta1):
        raise ValueError("need 0 <= beta0 <= log10(beta1), so that the schedule increases")
    k = anneal_steps // 3
    return np.concatenate([
        np.linspace(beta0, np.log10(beta1), k),
        np.linspace(np.log10(beta1), beta1, anneal_steps - k),
    ])


# meta-heuristic method for solving an optimization problem.
# One needs to define:
# - probl.init_config()
# - probl.cost()
# - probl.copy()
# - move = probl.propose_move()
# - delta_c = probl.compute_delta_cost(move)
# - probl.accept_move(move)
def simann(probl, beta0=0.1, beta1=10., anneal_steps=10, mcmc_steps=10, seed=None,
           wait=10, wait_2=30, betas=None, plot=False, verbose=True, return_history=False):
    """
    Inputs:
        -probl: the problem to solve (e.g. a Grid)
        -beta0: the minimum beta value (maximum temperature)
        -beta1: the maximum beta value (minimum temperature)
        -anneal_steps: number of beta values between beta0 and beta1
        -mcmc_steps: Markov Chain Monte Carlo steps for each beta value
        -seed: random seed. The starting configuration is drawn after seeding,
               so a run is fully determined by the problem and the seed
        -wait: halt after this many consecutive beta values in which the walker
               never moved (None disables this rule)
        -wait_2: halt after this many consecutive beta values without improving
                 the best cost (None disables this rule)
        -betas: optional custom schedule that replaces beta0, beta1 and
                anneal_steps, e.g. the logarithmic schedule of the report,
                np.log(2 + np.arange(T)) / c, together with mcmc_steps=1
        -plot: plot the frequency of accepted moves against beta
        -verbose: print the best cost at the end
        -return_history: also return a dict with statistics for each beta
    Returns best_probl, best_c (and history if return_history is True), where
    best_probl is a copy of the problem at the best configuration visited.

    Each step proposes a move with probl.propose_move() and accepts it with the
    Metropolis rule, using the cost change of exactly the move that was proposed.
    A proposed null move (0, 0) leaves the walker where it is.
    """
    if seed is not None:
        np.random.seed(seed)
    probl.init_config()
    if betas is None:
        betas = two_segment_schedule(beta0, beta1, anneal_steps)

    best_c = probl.cost()
    best_probl = probl.copy()
    patience, stale = wait, 0
    history = {"beta": [], "accept_freq": [], "cost": [], "best_cost": []}

    for beta in betas:
        accepted, improved = 0, False
        for _ in range(mcmc_steps):
            move = probl.propose_move()
            if tuple(move) == (0, 0):  # null move: the walker stays where it is
                continue
            delta_c = probl.compute_delta_cost(move)
            if accept_with_prob(delta_c, beta):
                probl.accept_move(move)
                accepted += 1
                c = probl.cost()
                if c < best_c:
                    best_c, best_probl, improved = c, probl.copy(), True

        history["beta"].append(beta)
        history["accept_freq"].append(accepted / mcmc_steps)
        history["cost"].append(probl.cost())
        history["best_cost"].append(best_c)

        if wait is not None:
            patience = wait if accepted > 0 else patience - 1
            if patience <= 0:
                break
        if wait_2 is not None:
            stale = 0 if improved else stale + 1
            if stale >= wait_2:
                break

    if verbose:
        print(f"best cost = {best_c}")
    if plot:
        plt.plot(history["beta"], history["accept_freq"], marker=".")
        plt.xlabel("beta")
        plt.ylabel("frequency of accepted moves")
    if return_history:
        return best_probl, best_c, history
    return best_probl, best_c
