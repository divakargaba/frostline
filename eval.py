"""Evaluation harness — baselines vs agent.

Owner: Physics+Eval

Compares the hydrate-detection agent against simple baselines
(threshold-only, ML-only, physics-only) on the 3W test set.

Metrics:
  - Lead time (minutes before established-hydrate phase)
  - False alarms per day
  - Misdiagnosis rate (wrong event class)

TODO:
  - Implement baseline runners
  - Implement agent runner with replay
  - Compute and tabulate metrics
  - Generate comparison plots
"""

import pandas as pd


def run_baseline_threshold(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run a simple threshold-based baseline on the test set.

    Args:
        test_df: Test dataframe with sensor readings and labels.

    Returns:
        DataFrame with columns [well_id, timestamp, prediction, label].
    """
    raise NotImplementedError


def run_baseline_ml_only(test_df: pd.DataFrame) -> pd.DataFrame:
    """Run ML-only baseline (no physics, no agent) on the test set.

    Args:
        test_df: Test dataframe with sensor readings and labels.

    Returns:
        DataFrame with columns [well_id, timestamp, prediction, label].
    """
    raise NotImplementedError


def run_agent(test_df: pd.DataFrame) -> pd.DataFrame:
    """Replay the full agent pipeline on the test set.

    Args:
        test_df: Test dataframe with sensor readings and labels.

    Returns:
        DataFrame with columns [well_id, timestamp, decision, label, tool_calls].
    """
    raise NotImplementedError


def compute_metrics(results_df: pd.DataFrame) -> dict:
    """Compute lead time, false alarms/day, and misdiagnosis rate.

    Args:
        results_df: Results from a baseline or agent run.

    Returns:
        Dict with keys: lead_time_min, false_alarms_per_day, misdiagnosis_rate.
    """
    raise NotImplementedError


def main() -> None:
    """Run all baselines and the agent, then print comparison table."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
