# MIP-Based Fair Goal Assignment Algorithm

## Overview

This document describes the Mixed Integer Programming (MIP) algorithm for fair goal assignment in multi-robot formation control. The algorithm optimizes a permutation matrix **P** to minimize the **maximum agent cost** (minimax objective), ensuring fairness across all agents.

## Problem Formulation

### Objective
Find a permutation matrix **P** ∈ {0,1}<sup>N×N</sup> that minimizes:
```
max_i (cost_i)
```
where `cost_i` is the total game cost for agent `i` when assigned to goal `P @ g[i]`.

### Constraints
- **Row constraints**: Each agent assigned to exactly one goal
  ```
  Σ_j P[i,j] = 1  for all i ∈ {1, ..., N}
  ```
- **Column constraints**: Each goal assigned to exactly one agent
  ```
  Σ_i P[i,j] = 1  for all j ∈ {1, ..., N}
  ```

## Algorithm Pseudocode

### Main Optimization Loop

```
Algorithm: MIP Fair Goal Assignment
Input: 
  - N: number of agents/goals
  - x0: initial agent positions (N, 2)
  - g: goal positions (N, 2)
  - game_solver: game-theoretic solver instance
  - num_iters: number of MIP iterations
Output: 
  - P*: optimal permutation matrix (N, N)
  - hat_g: assigned goals (N, 2) = P* @ g

1. // Initialize with Hungarian assignment (baseline)
   dist_matrix ← compute_pairwise_distances(x0, g)  // (N, N)
   P_init ← Hungarian_Assignment(dist_matrix)
   costs_init ← Evaluate_Game_Costs(P_init, game_solver)
   best_P ← P_init
   best_max_cost ← max(costs_init)
   
2. // Initialize evaluation pool
   evaluated_pool ← {}
   evaluated_pool[P_to_key(P_init)] ← (P_init, costs_init, max(costs_init))
   
3. // Iterative MIP refinement
   FOR iteration = 1 TO num_iters:
     
     a. // Update cost matrix from evaluated pool
        cost_matrix ← Update_Cost_Matrix(evaluated_pool, dist_matrix)
        
     b. // Solve MIP to find candidate assignment
        P_candidate ← MIP_Optimize_Minimax(cost_matrix)
        
     c. // Skip if already evaluated
        IF P_candidate ∈ evaluated_pool:
           CONTINUE
        
     d. // Evaluate candidate via game solver
        costs_candidate ← Evaluate_Game_Costs(P_candidate, game_solver)
        max_cost_candidate ← max(costs_candidate)
        
     e. // Add to pool
        evaluated_pool[P_to_key(P_candidate)] ← 
            (P_candidate, costs_candidate, max_cost_candidate)
        
     f. // Update best if improved
        IF max_cost_candidate < best_max_cost:
           best_P ← P_candidate
           best_max_cost ← max_cost_candidate
   
4. RETURN best_P, best_P @ g
```

### MIP Optimization Subroutine

```
Algorithm: MIP_Optimize_Minimax
Input: 
  - cost_matrix: (N, N) estimated costs for each agent-goal pair
Output: 
  - P: (N, N) permutation matrix

1. // Variables: P[i,j] ∈ {0,1} for all i,j, plus z ∈ ℝ (for minimax)
   n_vars ← N*N + 1
   
2. // Objective: minimize z (the maximum cost)
   c ← [0, ..., 0, 1]  // Last element is z
   
3. // Variable bounds
   integrality ← [1, ..., 1, 0]  // P binary, z continuous
   bounds.lb ← [0, ..., 0, 0]
   bounds.ub ← [1, ..., 1, ∞]
   
4. // Constraints
   A ← zeros((2*N + N, n_vars))
   b_lower ← zeros(2*N + N)
   b_upper ← zeros(2*N + N)
   
   constraint_idx ← 0
   
   // Row constraints: Σ_j P[i,j] = 1
   FOR i = 1 TO N:
      FOR j = 1 TO N:
         A[constraint_idx, i*N + j] ← 1.0
      b_lower[constraint_idx] ← 1.0
      b_upper[constraint_idx] ← 1.0
      constraint_idx ← constraint_idx + 1
   
   // Column constraints: Σ_i P[i,j] = 1
   FOR j = 1 TO N:
      FOR i = 1 TO N:
         A[constraint_idx, i*N + j] ← 1.0
      b_lower[constraint_idx] ← 1.0
      b_upper[constraint_idx] ← 1.0
      constraint_idx ← constraint_idx + 1
   
   // Minimax constraints: Σ_j P[i,j] * cost_matrix[i,j] ≤ z
   FOR i = 1 TO N:
      FOR j = 1 TO N:
         A[constraint_idx, i*N + j] ← cost_matrix[i, j]
      A[constraint_idx, -1] ← -1.0  // -z
      b_lower[constraint_idx] ← -∞
      b_upper[constraint_idx] ← 0.0
      constraint_idx ← constraint_idx + 1
   
5. // Solve MIP
   result ← MILP_Solve(c, A, b_lower, b_upper, bounds, integrality)
   
6. // Extract P from solution
   P_flat ← result.x[:-1]  // Exclude z
   P ← reshape(P_flat, (N, N))
   P ← round(P)  // Ensure binary
   
7. RETURN P
```

### Cost Matrix Update Subroutine

```
Algorithm: Update_Cost_Matrix
Input: 
  - evaluated_pool: dictionary of evaluated assignments
  - dist_matrix: (N, N) distance matrix
Output: 
  - cost_matrix: (N, N) updated cost estimates

1. cost_counts ← zeros((N, N))
   cost_matrix_updated ← zeros((N, N))
   
2. // Average costs from evaluated assignments
   FOR each (key, (P_eval, costs_eval, _)) IN evaluated_pool:
      FOR i = 1 TO N:
         j ← argmax(P_eval[i, :])  // Goal assigned to agent i
         IF cost_counts[i, j] == 0:
            cost_matrix_updated[i, j] ← costs_eval[i]
         ELSE:
            // Running average
            cost_matrix_updated[i, j] ← 
               (cost_matrix_updated[i, j] * cost_counts[i, j] + costs_eval[i]) 
               / (cost_counts[i, j] + 1)
         cost_counts[i, j] ← cost_counts[i, j] + 1
   
3. // Fill unevaluated pairs with distance-based estimates
   FOR i = 1 TO N:
      FOR j = 1 TO N:
         IF cost_counts[i, j] == 0:
            // Scale distance by average cost/distance ratio from pool
            IF evaluated_pool is not empty:
               ratios ← []
               FOR each (_, (P_eval, costs_eval, _)) IN evaluated_pool:
                  FOR k = 1 TO N:
                     g_idx ← argmax(P_eval[k, :])
                     dist_k ← dist_matrix[k, g_idx]
                     IF dist_k > ε:
                        ratios.append(costs_eval[k] / dist_k)
               IF ratios is not empty:
                  avg_ratio ← mean(ratios)
                  cost_matrix_updated[i, j] ← dist_matrix[i, j] * avg_ratio
               ELSE:
                  cost_matrix_updated[i, j] ← 
                     dist_matrix[i, j] * (best_max_cost / max(dist_matrix))
            ELSE:
               cost_matrix_updated[i, j] ← dist_matrix[i, j]
   
4. RETURN cost_matrix_updated
```

### Game Cost Evaluation Subroutine

```
Algorithm: Evaluate_Game_Costs
Input: 
  - P: (N, N) permutation matrix
  - game_solver: game-theoretic solver instance
Output: 
  - costs: (N,) array of total costs per agent

1. // Compute assigned goals
   hat_g ← P @ g  // (N, 2)
   
2. // Update game model with new assignment
   game_model["hat_g"] ← hat_g
   Rebuild_Runtime_Losses(game_model, hat_g)  // Update loss functions
   
3. // Temporarily set P in solver
   game_solver.params["P"] ← P
   game_solver.params["assignment_model"] ← None
   
4. // Solve game (with reduced iterations for speed)
   original_num_iters ← game_solver.params["num_iters"]
   game_solver.params["num_iters"] ← 10
   game_solver.solve_game()
   
5. // Extract agent costs
   x_traj_list ← game_solver.solution["x_traj_list"]
   u_traj_list ← game_solver.solution["u_traj_list"]
   costs ← game_solver.compute_agent_costs(x_traj_list, u_traj_list, hat_g)
   
6. // Restore original settings
   game_solver.params["num_iters"] ← original_num_iters
   game_solver.params.pop("P", None)
   
7. // Handle failures
   IF solve failed OR costs contains NaN/Inf:
      RETURN [1e6, ..., 1e6]  // High penalty
   
8. RETURN costs
```

## Key Features

### 1. **Minimax Fairness**
The algorithm minimizes the maximum agent cost, ensuring no single agent bears an unfairly high burden.

### 2. **Iterative Refinement**
- Starts with Hungarian assignment (minimizes total distance)
- Iteratively refines cost estimates by evaluating candidate assignments
- Uses MIP to propose new assignments based on refined cost estimates

### 3. **Cost Matrix Learning**
- Maintains a pool of evaluated assignments
- Updates cost estimates by averaging actual game costs
- Uses distance-based estimates for unevaluated agent-goal pairs

### 4. **Efficient Evaluation**
- Caches evaluated assignments to avoid redundant game solves
- Uses reduced solver iterations (10 instead of 20) for faster evaluation
- Skips diagnostic checks during inner optimization

## Complexity

- **MIP Solver**: O(N²) variables, O(N) constraints → typically polynomial in practice
- **Game Evaluation**: O(N × T × iterations) per evaluation, where T is time horizon
- **Overall**: O(num_iters × (MIP_time + game_solve_time))

## Implementation Notes

1. **Fallback to Hungarian**: If MIP solver is unavailable or fails, falls back to Hungarian algorithm on the cost matrix.

2. **Numerical Stability**: 
   - Handles NaN/Inf in costs by returning high penalties
   - Uses safe normalization in distance computations
   - Clips and validates all intermediate values

3. **Permutation Representation**: 
   - Uses tuple of goal indices as dictionary keys for fast lookup
   - Converts between permutation matrix and tuple representation efficiently

## Example Usage

```python
# Initialize optimizer
optimizer = FairAssignmentOptimizer(
    N=10,
    game_solver=gs
)

# Optimize assignment
optimizer.optimize(num_iters=20, verbose=True)

# Get optimal assignment
P_optimal = optimizer.P()  # (10, 10) permutation matrix
hat_g = optimizer.hat_g(g)  # (10, 2) assigned goals
```

## References

- **MIP Formulation**: Standard assignment problem with minimax objective
- **Game Solver**: Uses iLQR/LQRAX-style iterative solver for trajectory optimization
- **Cost Function**: Weighted combination of goal tracking, collision avoidance, and control effort

