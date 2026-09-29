from cosched.cli import run_simulation, run_horizon_sweep


def test_cli_simulation():
    res = run_simulation(policy="P0", num_workflows=5)
    assert res["policy"] == "P0"
    assert res["completed_workflows"] == 5


def test_cli_horizon_sweep():
    results = run_horizon_sweep(num_workflows=5)
    assert len(results) == 5
    assert results[0]["horizon_sec"] == 1.0
