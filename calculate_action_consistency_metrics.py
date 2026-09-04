"""
Script for calculating action consistency metrics

Usage:
python calculate_action_consistency_metrics.py --result_dir /path/to/results

Note: This script requires the modified logger to save detailed per-step information
"""

import os
import json
import pickle
import numpy as np
import argparse
from typing import Dict, List, Tuple

# Set matplotlib backend to avoid GUI-related issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for handling numpy data types
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def calculate_direction_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Calculate the direction change angle between three consecutive points
    
    Args:
        p1, p2, p3: Three consecutive position points (x, y)
    
    Returns:
        Direction change angle (degrees, 0-180)
    """
    # Calculate two vectors
    v1 = p2 - p1
    v2 = p3 - p2
    
    # Handle zero vector cases
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # Calculate angle (radians)
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Prevent numerical error
    angle_rad = np.arccos(cos_angle)
    
    # Convert to degrees
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def calculate_direction_changes(trajectory: np.ndarray) -> List[float]:
    """
    Calculate the list of direction change angles for the entire trajectory
    
    Args:
        trajectory: (N, 2) array of N position points
    
    Returns:
        List of angles with length N-2
    """
    if len(trajectory) < 3:
        return []
    
    angles = []
    for i in range(len(trajectory) - 2):
        angle = calculate_direction_angle(
            trajectory[i], 
            trajectory[i+1], 
            trajectory[i+2]
        )
        angles.append(angle)
    
    return angles


def calculate_stability_ratio(angles: List[float], threshold_deg: float = 30.0) -> float:
    """
    Calculate the direction stability ratio (fraction of angle changes less than or equal to threshold)
    
    Args:
        angles: List of direction change angles
        threshold_deg: Angle threshold (degrees)
    
    Returns:
        Stability ratio [0, 1]
    """
    if len(angles) == 0:
        return 0.0
    
    stable_count = sum(1 for angle in angles if angle <= threshold_deg)
    return stable_count / len(angles)


def calculate_frontier_reselection_rate(frontier_choices: List[str]) -> Tuple[float, Dict]:
    """
    Calculate the frontier reselection rate
    
    Args:
        frontier_choices: List of selected frontier IDs at each step
    
    Returns:
        (reselection_rate, choice_counts_dict)
    """
    if len(frontier_choices) == 0:
        return 0.0, {}
    
    # Count selection occurrences for each frontier
    choice_counts = {}
    for choice in frontier_choices:
        choice_counts[choice] = choice_counts.get(choice, 0) + 1
    
    # Calculate the fraction of reselected frontiers
    reselected = sum(1 for count in choice_counts.values() if count > 1)
    reselection_rate = reselected / len(choice_counts) if len(choice_counts) > 0 else 0.0
    
    return reselection_rate, choice_counts


def calculate_position_revisit_rate(trajectory: np.ndarray, distance_threshold: float = 0.5) -> Tuple[float, List[int]]:
    """
    Calculate position revisit rate (detect whether the agent revisits the same or nearby positions)
    
    Args:
        trajectory: (N, 2) array of N position points
        distance_threshold: Distance threshold (meters), default 0.5m
    
    Returns:
        (revisit_rate, revisit_counts_per_step)
    """
    if len(trajectory) < 2:
        return 0.0, []
    
    revisit_counts = []
    
    # For each position (starting from the 2nd), check its distance to all previous positions
    for i in range(1, len(trajectory)):
        current_pos = trajectory[i]
        previous_positions = trajectory[:i]
        
        # Calculate distance between current position and all previous positions
        distances = np.linalg.norm(previous_positions - current_pos, axis=1)
        
        # Count positions within the distance threshold (revisit count)
        revisit_count = np.sum(distances <= distance_threshold)
        revisit_counts.append(revisit_count)
    
    # Calculate revisit rate: fraction of steps with revisits
    revisit_rate = np.sum(np.array(revisit_counts) > 0) / len(revisit_counts) if len(revisit_counts) > 0 else 0.0
    
    return revisit_rate, revisit_counts


def calculate_position_clustering(trajectory: np.ndarray, distance_threshold: float = 0.5) -> Dict:
    """
    Calculate position clustering statistics (analyze how many distinct areas the agent acts in)
    
    Args:
        trajectory: (N, 2) array of N position points
        distance_threshold: Distance threshold (meters)
    
    Returns:
        Clustering statistics dictionary
    """
    if len(trajectory) < 2:
        return {"n_clusters": 1, "mean_cluster_size": 1, "max_cluster_size": 1}
    
    # Simple clustering: if distance between two points is less than threshold, consider them in the same area
    visited_clusters = []  # Each element is a cluster center
    cluster_sizes = []  # Size of each cluster
    
    for pos in trajectory:
        # Check if belongs to an existing cluster
        found_cluster = False
        for i, cluster_center in enumerate(visited_clusters):
            if np.linalg.norm(pos - cluster_center) <= distance_threshold:
                cluster_sizes[i] += 1
                found_cluster = True
                break
        
        # If not belonging to any existing cluster, create new cluster
        if not found_cluster:
            visited_clusters.append(pos)
            cluster_sizes.append(1)
    
    return {
        "n_clusters": len(visited_clusters),
        "mean_cluster_size": np.mean(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "cluster_distribution": cluster_sizes
    }


def analyze_single_episode(episode_data: Dict, distance_threshold: float = 0.5) -> Dict:
    """
    Analyze action consistency metrics for a single episode
    
    Args:
        episode_data: Dictionary containing trajectory and selection information
            - "trajectory": (N, 2) position sequence
            - "frontier_choices": frontier selection sequence (optional)
        distance_threshold: Distance threshold for position revisits (meters)
    
    Returns:
        Metrics dictionary
    """
    trajectory = np.array(episode_data["trajectory"])
    
    # Metric 1: Direction change angle
    angles = calculate_direction_changes(trajectory)
    
    metrics = {
        "n_steps": len(trajectory),
        "direction_changes": angles,
        "mean_angle_change": np.mean(angles) if len(angles) > 0 else 0.0,
        "std_angle_change": np.std(angles) if len(angles) > 0 else 0.0,
        "max_angle_change": np.max(angles) if len(angles) > 0 else 0.0,
    }
    
    # Metric 2: Direction stability (multiple thresholds)
    for threshold in [30, 45, 60, 90]:
        stability = calculate_stability_ratio(angles, threshold)
        metrics[f"stability_ratio_{threshold}deg"] = stability
    
    # Metric 3: Frontier reselection rate (if data is available)
    if "frontier_choices" in episode_data and episode_data["frontier_choices"]:
        reselection_rate, choice_counts = calculate_frontier_reselection_rate(
            episode_data["frontier_choices"]
        )
        metrics["frontier_reselection_rate"] = reselection_rate
        metrics["unique_frontiers_selected"] = len(choice_counts)
        metrics["total_frontier_selections"] = len(episode_data["frontier_choices"])
    
    # Metric 4: Position revisit rate
    revisit_rate, revisit_counts = calculate_position_revisit_rate(trajectory, distance_threshold)
    metrics["position_revisit_rate"] = revisit_rate
    metrics["mean_revisit_count"] = np.mean(revisit_counts) if len(revisit_counts) > 0 else 0.0
    metrics["max_revisit_count"] = max(revisit_counts) if len(revisit_counts) > 0 else 0
    metrics["total_revisits"] = sum(revisit_counts) if len(revisit_counts) > 0 else 0
    
    # Metric 5: Position clustering statistics (analyze distribution of explored areas)
    clustering_stats = calculate_position_clustering(trajectory, distance_threshold)
    metrics["n_position_clusters"] = clustering_stats["n_clusters"]
    metrics["mean_cluster_size"] = clustering_stats["mean_cluster_size"]
    metrics["max_cluster_size"] = clustering_stats["max_cluster_size"]
    
    return metrics


def load_episode_data(episode_dir: str) -> Dict:
    """
    Load data from an episode directory
    
    Required file formats:
    - trajectory.json: {"positions": [[x1, y1], [x2, y2], ...]}
    - frontier_choices.json: {"choices": ["frontier_0", "frontier_1", ...]}
    """
    trajectory_path = os.path.join(episode_dir, "trajectory.json")
    frontier_path = os.path.join(episode_dir, "frontier_choices.json")
    
    episode_data = {}
    
    # Load trajectory data
    if os.path.exists(trajectory_path):
        with open(trajectory_path, 'r') as f:
            data = json.load(f)
            episode_data["trajectory"] = data["positions"]
    else:
        print(f"Warning: {trajectory_path} not found")
        return None
    
    # Load frontier selection data (optional)
    if os.path.exists(frontier_path):
        with open(frontier_path, 'r') as f:
            data = json.load(f)
            episode_data["frontier_choices"] = data.get("choices", [])
    
    return episode_data


def analyze_all_episodes(result_dir: str, distance_threshold: float = 0.5) -> Dict:
    """
    Analyze all episodes and summarize statistics
    """
    exp_dir = os.path.join(result_dir, "exp_eval_aeqa")
    
    if not os.path.exists(exp_dir):
        print(f"Error: {exp_dir} not found")
        return None
    
    # Get all episode directories
    episode_dirs = [
        os.path.join(exp_dir, d) 
        for d in os.listdir(exp_dir) 
        if os.path.isdir(os.path.join(exp_dir, d))
    ]
    
    all_metrics = []
    failed_episodes = []
    
    for episode_dir in episode_dirs:
        episode_id = os.path.basename(episode_dir)
        episode_data = load_episode_data(episode_dir)
        
        if episode_data is None:
            failed_episodes.append(episode_id)
            continue
        
        metrics = analyze_single_episode(episode_data, distance_threshold)
        metrics["episode_id"] = episode_id
        all_metrics.append(metrics)
    
    if len(all_metrics) == 0:
        print("Error: No valid episode data found")
        print("Please make sure the code was modified to save trajectory.json and frontier_choices.json")
        return None
    
    # Summary statistics
    summary = {
        "n_episodes": len(all_metrics),
        "n_failed": len(failed_episodes),
        "mean_angle_change": np.mean([m["mean_angle_change"] for m in all_metrics]),
        "std_angle_change": np.std([m["mean_angle_change"] for m in all_metrics]),
    }
    
    # Average stability under each threshold
    for threshold in [30, 45, 60, 90]:
        key = f"stability_ratio_{threshold}deg"
        values = [m[key] for m in all_metrics]
        summary[f"mean_{key}"] = np.mean(values)
        summary[f"std_{key}"] = np.std(values)
    
    # Position revisit rate statistics
    summary["mean_position_revisit_rate"] = np.mean([m["position_revisit_rate"] for m in all_metrics])
    summary["std_position_revisit_rate"] = np.std([m["position_revisit_rate"] for m in all_metrics])
    summary["mean_revisit_count"] = np.mean([m["mean_revisit_count"] for m in all_metrics])
    summary["mean_total_revisits"] = np.mean([m["total_revisits"] for m in all_metrics])
    
    # Position clustering statistics
    summary["mean_n_position_clusters"] = np.mean([m["n_position_clusters"] for m in all_metrics])
    summary["mean_cluster_size"] = np.mean([m["mean_cluster_size"] for m in all_metrics])
    
    # Frontier-related statistics (if data is available)
    frontier_metrics = [m for m in all_metrics if "frontier_reselection_rate" in m]
    if len(frontier_metrics) > 0:
        summary["mean_frontier_reselection_rate"] = np.mean(
            [m["frontier_reselection_rate"] for m in frontier_metrics]
        )
        summary["n_episodes_with_frontier_data"] = len(frontier_metrics)
        summary["frontier_data_coverage"] = len(frontier_metrics) / len(all_metrics)
    
    return {
        "summary": summary,
        "per_episode": all_metrics,
        "failed_episodes": failed_episodes
    }


def visualize_results(results: Dict, output_dir: str):
    """
    Visualize results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    per_episode = results["per_episode"]
    
    # 1. Direction change angle distribution
    all_angles = []
    for episode in per_episode:
        all_angles.extend(episode["direction_changes"])
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_angles, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Direction Change Angle (degrees)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Direction Changes')
    plt.axvline(np.mean(all_angles), color='r', linestyle='--', 
                label=f'Mean: {np.mean(all_angles):.2f}°')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'angle_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mean angle change per episode
    mean_angles = [m["mean_angle_change"] for m in per_episode]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(mean_angles)), mean_angles, alpha=0.7)
    plt.xlabel('Episode Index')
    plt.ylabel('Mean Direction Change (degrees)')
    plt.title('Mean Direction Change per Episode')
    plt.axhline(np.mean(mean_angles), color='r', linestyle='--', 
                label=f'Overall Mean: {np.mean(mean_angles):.2f}°')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'per_episode_angles.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Stability ratio comparison (different thresholds)
    thresholds = [30, 45, 60, 90]
    stability_means = [
        results["summary"][f"mean_stability_ratio_{t}deg"] for t in thresholds
    ]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, stability_means, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Angle Threshold (degrees)')
    plt.ylabel('Stability Ratio')
    plt.title('Direction Stability at Different Thresholds')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    for t, s in zip(thresholds, stability_means):
        plt.text(t, s + 0.02, f'{s:.3f}', ha='center')
    plt.savefig(os.path.join(output_dir, 'stability_thresholds.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Position revisit rate analysis
    revisit_rates = [m["position_revisit_rate"] for m in per_episode]
    revisit_counts = [m["mean_revisit_count"] for m in per_episode]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 4.1 Revisit rate distribution
    ax1.hist(revisit_rates, bins=20, edgecolor='black', alpha=0.7, color='coral')
    ax1.set_xlabel('Position Revisit Rate')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Position Revisit Rates')
    ax1.axvline(np.mean(revisit_rates), color='r', linestyle='--', 
                label=f'Mean: {np.mean(revisit_rates):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 4.2 Revisit rate per episode
    ax2.bar(range(len(revisit_rates)), revisit_rates, alpha=0.7, color='coral')
    ax2.set_xlabel('Episode Index')
    ax2.set_ylabel('Position Revisit Rate')
    ax2.set_title('Position Revisit Rate per Episode')
    ax2.axhline(np.mean(revisit_rates), color='r', linestyle='--', 
                label=f'Mean: {np.mean(revisit_rates):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_revisit_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Exploration efficiency analysis: number of clusters vs steps
    n_steps = [m["n_steps"] for m in per_episode]
    n_clusters = [m["n_position_clusters"] for m in per_episode]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(n_steps, n_clusters, alpha=0.6, s=100)
    plt.xlabel('Number of Steps')
    plt.ylabel('Number of Position Clusters')
    plt.title('Exploration Efficiency: Unique Areas Visited vs Total Steps')
    
    # Add trend line
    if len(n_steps) > 1:
        z = np.polyfit(n_steps, n_clusters, 1)
        p = np.poly1d(z)
        plt.plot(sorted(n_steps), p(sorted(n_steps)), "r--", alpha=0.8, 
                label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'exploration_efficiency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization results saved to: {output_dir}")


def compare_methods(baseline_results: Dict, method_results: Dict) -> Dict:
    """
    Compare baseline and your method
    """
    comparison = {
        "metric": [],
        "baseline": [],
        "our_method": [],
        "improvement": []
    }
    
    # Compare mean direction change angle (lower is better)
    baseline_angle = baseline_results["summary"]["mean_angle_change"]
    method_angle = method_results["summary"]["mean_angle_change"]
    improvement = (baseline_angle - method_angle) / baseline_angle * 100
    
    comparison["metric"].append("Mean Direction Change (degrees)")
    comparison["baseline"].append(f"{baseline_angle:.2f}")
    comparison["our_method"].append(f"{method_angle:.2f}")
    comparison["improvement"].append(f"{improvement:.2f}%")
    
    # Compare stability ratio (higher is better)
    for threshold in [30, 45, 60, 90]:
        key = f"mean_stability_ratio_{threshold}deg"
        baseline_stab = baseline_results["summary"][key]
        method_stab = method_results["summary"][key]
        improvement = (method_stab - baseline_stab) / baseline_stab * 100
        
        comparison["metric"].append(f"Stability Ratio (<{threshold}°)")
        comparison["baseline"].append(f"{baseline_stab:.3f}")
        comparison["our_method"].append(f"{method_stab:.3f}")
        comparison["improvement"].append(f"{improvement:.2f}%")
    
    # Compare frontier reselection rate (if both have data)
    if "mean_frontier_reselection_rate" in baseline_results["summary"] and \
       "mean_frontier_reselection_rate" in method_results["summary"]:
        baseline_resel = baseline_results["summary"]["mean_frontier_reselection_rate"]
        method_resel = method_results["summary"]["mean_frontier_reselection_rate"]
        improvement = (baseline_resel - method_resel) / baseline_resel * 100
        
        comparison["metric"].append("Frontier Reselection Rate")
        comparison["baseline"].append(f"{baseline_resel:.3f}")
        comparison["our_method"].append(f"{method_resel:.3f}")
        comparison["improvement"].append(f"{improvement:.2f}%")
    
    # Compare position revisit rate (lower is better)
    baseline_revisit = baseline_results["summary"]["mean_position_revisit_rate"]
    method_revisit = method_results["summary"]["mean_position_revisit_rate"]
    improvement = (baseline_revisit - method_revisit) / baseline_revisit * 100
    
    comparison["metric"].append("Position Revisit Rate")
    comparison["baseline"].append(f"{baseline_revisit:.3f}")
    comparison["our_method"].append(f"{method_revisit:.3f}")
    comparison["improvement"].append(f"{improvement:.2f}%")
    
    # Compare number of position clusters (more indicates exploration of more distinct areas)
    baseline_clusters = baseline_results["summary"]["mean_n_position_clusters"]
    method_clusters = method_results["summary"]["mean_n_position_clusters"]
    improvement = (method_clusters - baseline_clusters) / baseline_clusters * 100
    
    comparison["metric"].append("Number of Unique Areas (Clusters)")
    comparison["baseline"].append(f"{baseline_clusters:.2f}")
    comparison["our_method"].append(f"{method_clusters:.2f}")
    comparison["improvement"].append(f"{improvement:.2f}%")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Calculate action consistency metrics")
    parser.add_argument("--result_dir", type=str, required=True,
                      help="Result directory path")
    parser.add_argument("--output_dir", type=str, default=None,
                      help="Output directory (defaults to result_dir/consistency_metrics)")
    parser.add_argument("--baseline_dir", type=str, default=None,
                      help="Baseline result directory (for comparison)")
    parser.add_argument("--distance_threshold", type=float, default=0.5,
                      help="Distance threshold for position revisits (meters), default 0.5m")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.result_dir, "consistency_metrics")
    
    print("="*60)
    print("Action Consistency Metrics Calculation")
    print("="*60)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Distance threshold: {args.distance_threshold}m")
    
    # Analyze current method
    print("\nAnalyzing current method results...")
    results = analyze_all_episodes(args.result_dir, args.distance_threshold)
    
    if results is None:
        print("\n" + "="*60)
        print("Error: Unable to load data")
        print("="*60)
        print("\nPlease make sure each episode directory contains the following files:")
        print("  - trajectory.json: Saves positions at each step")
        print("  - frontier_choices.json: Saves chosen frontier at each step (optional)")
        return
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    
    # Print summary
    print("\n" + "="*60)
    print("Statistical Summary")
    print("="*60)
    summary = results["summary"]
    print(f"Number of successfully analyzed episodes: {summary['n_episodes']}")
    print(f"Number of failed episodes: {summary['n_failed']}")
    print(f"\nMean direction change angle: {summary['mean_angle_change']:.2f}° ± {summary['std_angle_change']:.2f}°")
    
    print("\nDirection stability ratio (different thresholds):")
    for threshold in [30, 45, 60, 90]:
        mean_key = f"mean_stability_ratio_{threshold}deg"
        std_key = f"std_stability_ratio_{threshold}deg"
        print(f"  <{threshold}°: {summary[mean_key]:.3f} ± {summary[std_key]:.3f}")
    
    if "mean_frontier_reselection_rate" in summary:
        print(f"\nFrontier reselection rate:")
        print(f"  Mean: {summary['mean_frontier_reselection_rate']:.3f}")
        print(f"  Valid episodes: {summary['n_episodes_with_frontier_data']}/{summary['n_episodes']}")
        print(f"  Data coverage: {summary['frontier_data_coverage']*100:.1f}%")
    else:
        print(f"\n⚠️ Note: No episodes contain frontier selection data")
    
    # Position revisit rate statistics
    print(f"\nPosition revisit rate (distance threshold 0.5m):")
    print(f"  Mean revisit rate: {summary['mean_position_revisit_rate']:.3f} ({summary['mean_position_revisit_rate']*100:.1f}%)")
    print(f"  Standard deviation: {summary['std_position_revisit_rate']:.3f}")
    print(f"  Mean revisits per step: {summary['mean_revisit_count']:.2f}")
    print(f"  Mean total revisits: {summary['mean_total_revisits']:.2f}")
    
    print(f"\nPosition clustering statistics:")
    print(f"  Mean explored areas: {summary['mean_n_position_clusters']:.2f}")
    print(f"  Mean steps per area: {summary['mean_cluster_size']:.2f}")
    
    # Visualization
    print("\nGenerating visualization charts...")
    visualize_results(results, args.output_dir)
    
    # If baseline is provided, compare
    if args.baseline_dir:
        print("\nAnalyzing baseline results...")
        baseline_results = analyze_all_episodes(args.baseline_dir, args.distance_threshold)
        if baseline_results:
            comparison = compare_methods(baseline_results, results)
            
            print("\n" + "="*60)
            print("Method Comparison")
            print("="*60)
            for i in range(len(comparison["metric"])):
                print(f"{comparison['metric'][i]}")
                print(f"  Baseline: {comparison['baseline'][i]}")
                print(f"  Our Method: {comparison['our_method'][i]}")
                print(f"  Improvement: {comparison['improvement'][i]}")
                print()
            
            with open(os.path.join(args.output_dir, "comparison.json"), 'w') as f:
                json.dump(comparison, f, indent=4, cls=NumpyEncoder)
    
    print("\n" + "="*60)
    print(f"Analysis complete! Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
