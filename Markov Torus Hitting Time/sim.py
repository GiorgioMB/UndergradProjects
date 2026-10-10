import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
random.seed(42)
np.random.seed(42)

def get_diagonal_intersection(dx, dy, N, M):
    """
    Solves l = dx (mod N) and l = dy (mod M) for the minimal l > 0.
    Returns infinity if no solution exists.
    """
    g = math.gcd(N, M)
    if dx % g != dy % g:
        return float('inf')
        
    N_prime = N // g
    M_prime = M // g
    D = (dy - dx) // g
    
    if M_prime == 1:
        k = 0
    else:
        k = (int(D) * pow(int(N_prime), -1, int(M_prime))) % int(M_prime)

    l = dx + k * N
    return l if l > 0 else (N * M) // g

def simulate_fast_billiard_search(N, M, c=1.0):
    """
    O(1) step execution to bypass linear traversal limits.
    """
    cx, cy = random.randint(0, N - 1), random.randint(0, M - 1)
    tx, ty = random.randint(0, N - 1), random.randint(0, M - 1)
    
    if (cx, cy) == (tx, ty):
        return 0
        
    total_steps = 0
    j = 1
    
    while True:
        D_j = 2**j
        phase_budget = c * (D_j**2)
        phase_steps = 0
        
        while phase_steps < phase_budget:
            vx, vy = random.choice([(1,0), (0,1), (1,1), (1,-1)])
            L = random.randint(1, D_j)
            
            hit_l = float('inf')
            
            if vx == 1 and vy == 0:
                if cy == ty:
                    hit_l = (tx - cx) % N
                    if hit_l == 0: hit_l = N
            elif vx == 0 and vy == 1:
                if cx == tx:
                    hit_l = (ty - cy) % M
                    if hit_l == 0: hit_l = M
            elif vx == 1 and vy == 1:
                dx, dy = (tx - cx) % N, (ty - cy) % M
                hit_l = get_diagonal_intersection(dx, dy, N, M)
            elif vx == 1 and vy == -1:
                dx, dy = (tx - cx) % N, (cy - ty) % M
                hit_l = get_diagonal_intersection(dx, dy, N, M)
                
            if hit_l <= L:
                return total_steps + hit_l
                
            cx = (cx + vx * L) % N
            cy = (cy + vy * L) % M
            
            total_steps += L
            phase_steps += L
            
        j += 1

def run_experiments():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    N_vals = np.array([30, 50, 100, 150, 200, 300, 400, 500, 1000, 2000, 5000, 10000])
    R_vals = np.random.uniform(1, 10, size=len(N_vals))
    M_vals = np.round(N_vals * R_vals).astype(int)
    trials = 500
    results = []
    print(f"The sampled ratios R = M/N are: {R_vals}\n")
    for n, m in zip(N_vals, M_vals):
        S = n * m
        steps = [simulate_fast_billiard_search(n, m) for _ in range(trials)]
        print(f"Completed {trials} trials for state space size |S| = {S}")
        print(f"Min steps: {min(steps)}, Max steps: {max(steps)}, Mean steps: {np.mean(steps):.2f}, Std Dev: {np.std(steps):.2f}\n")
        results.append({
            'S': S,
            'mean': np.mean(steps),
            'std': np.std(steps)
        })
    S_array = np.array([r['S'] for r in results])
    mean_array = np.array([r['mean'] for r in results])
    std_array = np.array([r['std'] for r in results])
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(S_array, mean_array, marker='o', color='#2c3e50', linewidth=2, label=r'Empirical $\mathbb{E}[C_{x^*}]$')
    ax.fill_between(S_array, 
                    np.maximum(0, mean_array - std_array), 
                    mean_array + std_array, 
                    color='#34495e', alpha=0.2, label=r'$\pm 1$ Standard Deviation')
    
    # Fit a reference line y = k * S to highlight the linear bound
    k = mean_array[-1] / S_array[-1]
    ax.plot(S_array, k * S_array, linestyle='--', color='#e74c3c', linewidth=2, label=r'Linear Reference $\mathcal{O}(|S|)$')
    
    ax.set_xlabel(r'State Space Size $|S|$', fontsize=14)
    ax.set_ylabel(r'Total Steps', fontsize=14)
    ax.set_title('Hitting Time of Blind Ballistic Search on Torus', fontsize=16)
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig('hitting_time_plot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiments()

# %%
