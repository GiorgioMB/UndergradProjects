import zlib
from copy import copy as _shallow_copy

import numpy as np
import matplotlib.pyplot as plt

from generate_data import generate_data

# The nine moves of the proposal: the eight neighboring cells and the null move (0, 0).
# A move (dx, dy) changes the row index by dx and the column index by dy.
MOVES = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=np.int64)


def id_to_seed(individual_id):
    """Deterministic 32-bit seed derived from the ID string (the same on every run and machine)."""
    return zlib.crc32(individual_id.encode("utf-8"))


class Grid:
    """
    Grid problem for simulated annealing with the soft-max-biased proposal
    analyzed in the report (Sections 3 and 4).

    Input:
        - n: dimension of the problem
        - individual_id: your student ID
        - gamma: strength of the soft-max bias (gamma = 0 gives a uniform proposal)
        - np_seed: seed for the landscape values. By default it is derived from
          individual_id, so the landscape is a deterministic function of the ID
        - track_path: if True, the visited positions are recorded for display_end()
        Notes:
            -- n must be an integer, even and >68
            -- individual_id must be a string
        Failing to match these conditions will result in the class throwing an error

    The grid wraps around at its edges (indices are taken modulo n). From a cell x
    the proposal picks one of the nine moves with probability proportional to
    exp(-gamma * (f(y) - f(x))), where y is the target cell; this is the report's
    q_gamma(x, y) = exp(-gamma f(y)) / Z_gamma(x). Cost differences are computed
    on demand, so apart from the landscape the class stores only the current
    position: there is no visited matrix and no precomputed cost matrix.
    """
    def __init__(self, n, individual_id, gamma=0.05, np_seed=None, track_path=False):
        if type(n) != int:
            raise TypeError(f"{n} is invalid. Input number must be an integer")
        if n < 70 or n % 2 != 0:
            raise ValueError(f"{n} is invalid. Input number must be even and >68")
        if type(individual_id) != str:
            raise TypeError(f"{individual_id} is invalid. Input ID must be a string")
        if gamma < 0:
            raise ValueError(f"gamma = {gamma} is invalid. It must be non-negative")
        self.n = n
        self.individual_id = individual_id
        self.gamma = float(gamma)
        self.np_seed = id_to_seed(individual_id) if np_seed is None else np_seed
        # The landscape is generated once and never modified afterwards.
        self.f_values = generate_data(n, individual_id, np_seed=self.np_seed)
        self.track_path = track_path
        self.init_config()

    def name(self):
        return 'Grid (soft-max proposal)'

    def init_config(self):
        """Place the walker on a uniformly random cell."""
        self.position_x = int(np.random.randint(0, self.n))
        self.position_y = int(np.random.randint(0, self.n))
        self.path = [(self.position_x, self.position_y)] if self.track_path else None

    def copy(self):
        """
        Copy of the current state. The landscape array is shared rather than
        duplicated, since it is never modified, so copies are cheap.
        """
        new = _shallow_copy(self)
        if self.path is not None:
            new.path = list(self.path)
        return new

    def __repr__(self):
        return (f"Grid(n={self.n}, gamma={self.gamma}, "
                f"position=({self.position_x}, {self.position_y}), cost={self.cost():.4f})")

    def cost(self):
        return self.f_values[self.position_x, self.position_y]

    def neighbor_costs(self):
        """Costs of the nine cells reachable in one move (in the order of MOVES), computed on demand."""
        n = self.n
        return self.f_values[(self.position_x + MOVES[:, 0]) % n,
                             (self.position_y + MOVES[:, 1]) % n]

    def _proposal_weights(self):
        logits = -self.gamma * (self.neighbor_costs() - self.cost())
        return np.exp(logits - logits.max())  # shifting the logits avoids overflow

    def proposal_probabilities(self):
        """Soft-max proposal q_gamma(x, .) over the nine moves, in the order of MOVES."""
        weights = self._proposal_weights()
        return weights / weights.sum()

    def propose_move(self):
        """
        Draw a move (dx, dy) from the soft-max proposal. (0, 0) is the null move,
        which leaves the walker where it is.
        """
        cumulative = np.cumsum(self._proposal_weights())
        k = np.searchsorted(cumulative, np.random.random() * cumulative[-1], side="right")
        dx, dy = MOVES[min(k, len(MOVES) - 1)]
        return int(dx), int(dy)

    def compute_delta_cost(self, move):
        """Cost change f(new) - f(current) of the given move, computed on demand."""
        dx, dy = move
        n = self.n
        x, y = self.position_x, self.position_y
        return self.f_values[(x + dx) % n, (y + dy) % n] - self.f_values[x, y]

    def accept_move(self, move):
        """Move the walker, wrapping around at the edges of the grid."""
        dx, dy = move
        self.position_x = (self.position_x + dx) % self.n
        self.position_y = (self.position_y + dy) % self.n
        if self.path is not None:
            self.path.append((self.position_x, self.position_y))

    def display(self):
        """Show the landscape with the current position as a red dot."""
        self._show_landscape()
        plt.show()

    def display_beta(self, beta):
        self._show_landscape()
        plt.title(f"Beta = {beta}")
        plt.show()

    def display_end(self):
        """Show the landscape with the recorded path (requires track_path=True)."""
        if self.path is None:
            raise RuntimeError("No path was recorded: create the Grid with track_path=True")
        plt.imshow(self.f_values, cmap='viridis')
        plt.colorbar()
        path = np.array(self.path, dtype=float)
        # Break the line wherever the walker wraps around an edge of the grid.
        wraps = np.flatnonzero(np.abs(np.diff(path, axis=0)).max(axis=1) > 1) + 1
        path = np.insert(path, wraps, np.nan, axis=0)
        plt.plot(path[:, 1], path[:, 0], color='k', linewidth=1)
        plt.show()

    def _show_landscape(self):
        plt.imshow(self.f_values, cmap='viridis')
        plt.colorbar()
        plt.plot(self.position_y, self.position_x, 'ro')
