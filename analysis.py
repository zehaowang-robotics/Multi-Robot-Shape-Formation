#!/usr/bin/env python3
"""
Analysis script for multi-robot trajectory data.

Analyzes trajectory JSON files to compute:
1. Agent trajectory length (mean and max)
2. Agent collision counts (distance <= 0.3m between agents)
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_trajectories(json_file: str) -> Dict:
    """Load trajectory data from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def compute_trajectory_lengths(trajectories: Dict[str, List[Dict]]) -> Tuple[float, float]:
    """
    Compute mean and max trajectory length for all agents.
    
    Args:
        trajectories: Dictionary mapping agent_id to list of state dictionaries
        
    Returns:
        (mean_length, max_length): Mean and maximum trajectory lengths in meters
    """
    agent_lengths = []
    
    for agent_id, states in trajectories.items():
        if not states:
            continue
            
        # Sort by step to ensure correct order
        sorted_states = sorted(states, key=lambda s: s["step"])
        
        total_length = 0.0
        prev_x, prev_y = None, None
        
        for state in sorted_states:
            x, y = state["x"], state["y"]
            
            if prev_x is not None and prev_y is not None:
                # Compute Euclidean distance from previous position
                dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                total_length += dist
            
            prev_x, prev_y = x, y
        
        agent_lengths.append(total_length)
    
    if not agent_lengths:
        return 0.0, 0.0
    
    mean_length = np.mean(agent_lengths)
    max_length = np.max(agent_lengths)
    
    return mean_length, max_length


def compute_collision_counts(trajectories: Dict[str, List[Dict]], 
                             collision_threshold: float = 0.3) -> Dict[str, int]:
    """
    Count collisions between agents (distance <= threshold).
    
    Args:
        trajectories: Dictionary mapping agent_id to list of state dictionaries
        collision_threshold: Distance threshold in meters (default: 0.3m)
        
    Returns:
        Dictionary with:
            - "total_collisions": Total number of collision events
            - "collision_steps": Number of MPC steps with at least one collision
            - "agent_collisions": Dictionary mapping agent_id to collision count
    """
    # Organize states by step
    states_by_step = {}
    
    for agent_id, states in trajectories.items():
        for state in states:
            step = state["step"]
            if step < 0:  # Skip initial state (step -1)
                continue
                
            if step not in states_by_step:
                states_by_step[step] = {}
            states_by_step[step][int(agent_id)] = (state["x"], state["y"])
    
    total_collisions = 0
    collision_steps = set()
    agent_collisions = {agent_id: 0 for agent_id in trajectories.keys()}
    
    # Check collisions at each step
    for step, agent_positions in states_by_step.items():
        agent_ids = sorted(agent_positions.keys())
        step_has_collision = False
        
        # Check all pairs of agents
        for i, agent_i in enumerate(agent_ids):
            for agent_j in agent_ids[i+1:]:
                x_i, y_i = agent_positions[agent_i]
                x_j, y_j = agent_positions[agent_j]
                
                dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                
                if dist <= collision_threshold:
                    total_collisions += 1
                    step_has_collision = True
                    agent_collisions[str(agent_i)] += 1
                    agent_collisions[str(agent_j)] += 1
        
        if step_has_collision:
            collision_steps.add(step)
    
    return {
        "total_collisions": total_collisions,
        "collision_steps": len(collision_steps),
        "agent_collisions": agent_collisions,
        "max_agent_collisions": max(agent_collisions.values()) if agent_collisions else 0,
    }


def analyze_trajectories(json_file: str, collision_threshold: float = 0.3) -> Dict:
    """
    Perform complete analysis on trajectory file.
    
    Args:
        json_file: Path to trajectory JSON file
        collision_threshold: Distance threshold for collisions in meters
        
    Returns:
        Dictionary containing all analysis results
    """
    print(f"Loading trajectories from {json_file}...")
    data = load_trajectories(json_file)
    
    metadata = data.get("metadata", {})
    trajectories = data.get("trajectories", {})
    
    print(f"Analyzing {metadata.get('num_agents', 0)} agents over {metadata.get('num_steps', 0)} steps...")
    
    # Compute trajectory lengths
    mean_length, max_length = compute_trajectory_lengths(trajectories)
    
    # Compute collision counts
    collision_stats = compute_collision_counts(trajectories, collision_threshold)
    
    # Compile results
    results = {
        "file": json_file,
        "metadata": metadata,
        "trajectory_lengths": {
            "mean": float(mean_length),
            "max": float(max_length),
        },
        "collisions": {
            "threshold_m": collision_threshold,
            "total_collisions": collision_stats["total_collisions"],
            "collision_steps": collision_stats["collision_steps"],
            "max_agent_collisions": collision_stats["max_agent_collisions"],
            "agent_collisions": collision_stats["agent_collisions"],
        },
    }
    
    return results


def print_results(results: Dict):
    """Print analysis results in a readable format."""
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS RESULTS")
    print("="*60)
    
    metadata = results["metadata"]
    print(f"\nFile: {results['file']}")
    print(f"Assignment Method: {metadata.get('assignment_method', 'N/A')}")
    print(f"Goal Type: {metadata.get('goal_type', 'N/A')}")
    print(f"Formation Sequence: {metadata.get('formation_seq', 'N/A')}")
    print(f"Number of Agents: {metadata.get('num_agents', 'N/A')}")
    print(f"Number of Steps: {metadata.get('num_steps', 'N/A')}")
    print(f"Total Time: {metadata.get('total_time_seconds', 0):.2f} seconds")
    
    # Trajectory lengths
    print("\n" + "-"*60)
    print("TRAJECTORY LENGTHS")
    print("-"*60)
    lengths = results["trajectory_lengths"]
    print(f"Mean Trajectory Length: {lengths['mean']:.4f} m")
    print(f"Max Trajectory Length:  {lengths['max']:.4f} m")
    
    # Collisions
    print("\n" + "-"*60)
    print("COLLISION ANALYSIS")
    print("-"*60)
    collisions = results["collisions"]
    print(f"Collision Threshold: {collisions['threshold_m']:.2f} m")
    print(f"Total Collision Events: {collisions['total_collisions']}")
    print(f"Steps with Collisions: {collisions['collision_steps']}")
    print(f"Max Collisions per Agent: {collisions['max_agent_collisions']}")
    
    if collisions['total_collisions'] > 0:
        print("\nCollisions per Agent:")
        agent_collisions = collisions['agent_collisions']
        for agent_id in sorted(agent_collisions.keys(), key=int):
            count = agent_collisions[agent_id]
            if count > 0:
                print(f"  Agent {agent_id}: {count} collisions")
    
    print("\n" + "="*60)


def main():
    """Main entry point for analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze multi-robot trajectory performance from JSON files"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to trajectory JSON file to analyze"
    )
    parser.add_argument(
        "--collision-threshold",
        type=float,
        default=0.3,
        dest="collision_threshold",
        help="Distance threshold for collisions in meters (default: 0.3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        return 1
    
    # Perform analysis
    try:
        results = analyze_trajectories(args.json_file, args.collision_threshold)
        
        # Print results
        print_results(results)
        
        # Save to file if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

