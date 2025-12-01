# Goal Assignment Algorithms

This document describes the three goal assignment algorithms used in the multi-robot formation control system:
1. **Greedy Assignment** - Fast nearest-neighbor heuristic
2. **Hungarian Assignment** - Optimal total distance minimization
3. **MIP Fair Assignment** - Optimal minimax fairness (see `MIP_GOAL_ASSIGNMENT.md`)

## Problem Formulation

Given:
- **N agents** with initial positions `x0[i] ∈ ℝ²` for `i ∈ {0, ..., N-1}`
- **N goals** with positions `g[j] ∈ ℝ²` for `j ∈ {0, ..., N-1}`

Find a **permutation** (one-to-one assignment) that maps each agent to exactly one goal, producing assigned goals `hat_g[i]` for each agent `i`.

---

## 1. Greedy Assignment Algorithm

### Overview
The greedy algorithm assigns each agent to its nearest unassigned goal in a sequential manner. This is a fast heuristic that runs in O(N² log N) time but does not guarantee optimality.

### Objective
Minimize individual agent distances (greedy local optimization).

### Pseudocode

```
Algorithm: Greedy Assignment
Input: 
  - x0: (N, 2) agent initial positions
  - g: (N, 2) goal positions
Output: 
  - hat_g: (N, 2) assigned goals for each agent

1. N ← length(x0)
2. used ← empty set  // Track assigned goals
3. hat_g ← empty list
4. 
5. FOR i = 0 TO N-1:
   a. // Compute distances from agent i to all goals
      dists ← []
      FOR j = 0 TO N-1:
         dists[j] ← ||g[j] - x0[i]||  // Euclidean distance
      
   b. // Sort goals by distance (ascending)
      order ← argsort(dists)  // Indices sorted by distance
      
   c. // Find nearest unassigned goal
      chosen ← None
      FOR idx IN order:
         IF idx NOT IN used:
            chosen ← idx
            BREAK
      
   d. // Fallback: if all goals used (shouldn't happen), use nearest
      IF chosen == None:
         chosen ← order[0]
      
   e. // Assign goal to agent
      used.add(chosen)
      hat_g[i] ← g[chosen]
   
6. RETURN stack(hat_g)  // Convert to (N, 2) array
```

### Complexity
- **Time**: O(N² log N) - For each agent, compute N distances and sort
- **Space**: O(N) - Store used set and output

### Properties
- ✅ **Fast**: Simple and efficient
- ✅ **Deterministic**: Same input always produces same output
- ❌ **Not Optimal**: May produce suboptimal total distance
- ❌ **Order Dependent**: Result depends on agent processing order

### Example
```
Agents: A₁(0,0), A₂(1,0)
Goals:  G₁(2,0), G₂(3,0)

Step 1: A₁ → nearest is G₁ → assign A₁→G₁
Step 2: A₂ → nearest is G₁ (but used) → assign A₂→G₂
Result: Total distance = 2 + 2 = 4

Optimal (Hungarian): A₁→G₁, A₂→G₂ → Total = 2 + 2 = 4 (same in this case)
```

---

## 2. Hungarian Assignment Algorithm

### Overview
The Hungarian algorithm (also known as the Kuhn-Munkres algorithm) solves the assignment problem optimally. It finds the assignment that **minimizes the total distance** from all agents to their assigned goals.

### Objective
Minimize: `Σᵢ ||hat_g[i] - x0[i]||` (total distance)

### Pseudocode

```
Algorithm: Hungarian Assignment
Input: 
  - x0: (N, 2) agent initial positions
  - g: (N, 2) goal positions
Output: 
  - hat_g: (N, 2) assigned goals for each agent

1. N ← length(x0)
   
2. // Build cost matrix: cost[i, j] = distance from agent i to goal j
   cost_matrix ← zeros((N, N))
   FOR i = 0 TO N-1:
      FOR j = 0 TO N-1:
         cost_matrix[i, j] ← ||g[j] - x0[i]||  // Euclidean distance
   
   // Vectorized version (more efficient):
   // diff ← x0[:, None, :] - g[None, :, :]  // (N, N, 2)
   // cost_matrix ← ||diff|| along axis=2     // (N, N)
   
3. // Solve assignment problem using Hungarian algorithm
   // This finds row_indices and col_indices such that:
   //   - row_indices[i] is the agent index
   //   - col_indices[i] is the assigned goal index
   //   - The assignment minimizes Σᵢ cost_matrix[row_indices[i], col_indices[i]]
   (row_indices, col_indices) ← linear_sum_assignment(cost_matrix)
   
4. // Build assigned goals array
   hat_g ← zeros((N, 2))
   FOR k = 0 TO length(row_indices)-1:
      agent_idx ← row_indices[k]
      goal_idx ← col_indices[k]
      hat_g[agent_idx] ← g[goal_idx]
   
5. RETURN hat_g
```

### Hungarian Algorithm Details (linear_sum_assignment)

The `linear_sum_assignment` function (from `scipy.optimize`) implements the Hungarian algorithm:

```
Algorithm: Linear Sum Assignment (Hungarian Method)
Input: 
  - cost_matrix: (N, N) cost matrix
Output: 
  - row_indices: array of row (agent) indices
  - col_indices: array of column (goal) indices

// This is a standard implementation of the Hungarian algorithm:
// 1. Subtract row minima
// 2. Subtract column minima  
// 3. Cover zeros with minimum lines
// 4. If N lines needed → optimal assignment found
// 5. Otherwise, adjust costs and repeat

// For details, see:
// - Kuhn, H. W. (1955). "The Hungarian method for the assignment problem"
// - Munkres, J. (1957). "Algorithms for the Assignment and Transportation Problems"
```

### Complexity
- **Time**: O(N³) - Standard Hungarian algorithm complexity
- **Space**: O(N²) - Cost matrix storage

### Properties
- ✅ **Optimal**: Minimizes total distance
- ✅ **Deterministic**: Same input always produces same output
- ✅ **Efficient**: Polynomial time, well-optimized implementations available
- ❌ **Not Fair**: May assign some agents very far while others are close

### Example
```
Agents: A₁(0,0), A₂(10,0)
Goals:  G₁(1,0), G₂(11,0)

Cost Matrix:
        G₁  G₂
   A₁   1   11
   A₂   9   1

Hungarian: A₁→G₁ (cost=1), A₂→G₂ (cost=1) → Total = 2
Greedy:    A₁→G₁ (cost=1), A₂→G₁ (impossible, but if possible cost=9) → Total = 10

Hungarian finds optimal total distance = 2
```

---

## 3. Comparison Summary

| Algorithm | Objective | Complexity | Optimality | Fairness |
|-----------|-----------|-----------|------------|----------|
| **Greedy** | Minimize individual distances | O(N² log N) | ❌ Heuristic | ❌ Not guaranteed |
| **Hungarian** | Minimize total distance | O(N³) | ✅ Optimal | ❌ May be unfair |
| **MIP Fair** | Minimize max agent cost | O(iter × (MIP + game)) | ✅ Optimal | ✅ Fair (minimax) |

### When to Use

- **Greedy**: Fast prototyping, large N, when speed is critical
- **Hungarian**: When total distance matters, standard optimal assignment
- **MIP Fair**: When fairness is important, willing to pay computational cost

---

## Implementation Notes

### Distance Metric
All algorithms use **Euclidean distance** (L2 norm):
```
distance(a, b) = ||a - b|| = √((aₓ - bₓ)² + (aᵧ - bᵧ)²)
```

### Assignment Representation
The assignment is represented as:
- **Permutation matrix P**: `P[i, j] = 1` if agent `i` assigned to goal `j`, else `0`
- **Assigned goals**: `hat_g[i] = Σⱼ P[i, j] × g[j]` or directly `hat_g[i] = g[assigned_goal_idx[i]]`

### Integration with Game Solver
All assignment methods are called during:
1. **Initial game construction**: `GameSolver.construct_game()`
2. **MPC refresh**: `GameSolver.refresh_for_mpc(recompute_assignment=True)`
3. **Formation switching**: When goals change, assignment is recomputed

---

## References

1. **Greedy Algorithm**: Nearest-neighbor heuristic for assignment problems
2. **Hungarian Algorithm**: 
   - Kuhn, H. W. (1955). "The Hungarian method for the assignment problem". *Naval Research Logistics Quarterly*, 2(1-2), 83-97.
   - Munkres, J. (1957). "Algorithms for the Assignment and Transportation Problems". *Journal of the Society for Industrial and Applied Mathematics*, 5(1), 32-38.
3. **MIP Fair Assignment**: See `MIP_GOAL_ASSIGNMENT.md` for details

