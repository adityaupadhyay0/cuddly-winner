import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import List
from cosched.simulator.engine import SimulationEngine
from cosched.simulator.workloads import WorkloadGenerator


def run_simulation(policy: str, num_workflows: int = 20, num_gpus: int = 4, seed: int = 42) -> dict:
    gen = WorkloadGenerator(seed=seed)
    traces = [gen.generate_trace(num_steps=4) for _ in range(num_workflows)]
    engine = SimulationEngine(policy=policy, num_gpus=num_gpus)
    metrics = engine.run(traces)
    res = metrics.to_dict()
    res["policy"] = policy
    return res


def run_horizon_sweep(num_workflows: int = 20) -> List[dict]:
    horizons = [1.0, 2.0, 5.0, 10.0, 30.0]
    results = []
    gen = WorkloadGenerator(seed=42)
    traces = [gen.generate_trace(num_steps=4) for _ in range(num_workflows)]

    for h in horizons:
        engine = SimulationEngine(policy="P2", forecast_horizon=h, num_gpus=4)
        metrics = engine.run(traces)
        d = metrics.to_dict()
        d["horizon_sec"] = h
        results.append(d)
    return results


def save_experiment_results(experiment_name: str, matrix: List[dict]):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("experiments", "results", f"run-{experiment_name}-{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(matrix, f, indent=2)

    # Save CSV matrix
    if matrix:
        keys = list(matrix[0].keys())
        with open(os.path.join(output_dir, "matrix.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(matrix)

    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(prog="cosched", description="CoSched v0.3 CLI & Experiment Harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Run simulation for a policy")
    sim_parser.add_argument("--policy", type=str, default="P0", choices=["P0", "P1", "P2", "P3", "P4", "oracle"], help="Policy to simulate")
    sim_parser.add_argument("--workflows", type=int, default=20, help="Number of workflows")
    sim_parser.add_argument("--gpus", type=int, default=4, help="Number of GPUs")

    # Experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run experimental sweeps")
    exp_parser.add_argument("experiment_type", choices=["horizon-sweep", "error-sensitivity", "telemetry-delay", "oscillation", "all"])

    args = parser.parse_args()

    if args.command == "simulate":
        print(f"--- Running CoSched Simulation (Policy: {args.policy}) ---")
        res = run_simulation(policy=args.policy, num_workflows=args.workflows, num_gpus=args.gpus)
        print(json.dumps(res, indent=2))

    elif args.command == "experiment":
        print(f"--- Running Experiment Sweep: {args.experiment_type} ---")
        if args.experiment_type in ("horizon-sweep", "all"):
            matrix = run_horizon_sweep()
            save_experiment_results("horizon-sweep", matrix)
            print("Horizon Sweep Results:")
            print(json.dumps(matrix, indent=2))

        if args.experiment_type in ("all", "error-sensitivity"):
            policies = ["P0", "P1", "P2", "P3", "P4", "ORACLE"]
            matrix = [run_simulation(p) for p in policies]
            save_experiment_results("p0-p4-comparison", matrix)


if __name__ == "__main__":
    main()
