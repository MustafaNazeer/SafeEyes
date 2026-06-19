from safeeyes.temporal.tune_temporal import best_by, run_sweep


def test_run_sweep_evaluates_every_config_in_order() -> None:
    grid = [{"epochs": 1}, {"epochs": 2}]
    seen: list[dict[str, object]] = []

    def fake_evaluate(train: object, val: object, **cfg: object) -> dict[str, object]:
        seen.append(cfg)
        return {"macro_auroc": float(cfg["epochs"]) * 0.1}  # type: ignore[arg-type]

    results = run_sweep(["t"], ["v"], grid, evaluate=fake_evaluate)
    assert seen == [{"epochs": 1}, {"epochs": 2}]
    assert [cfg for cfg, _ in results] == grid
    assert results[1][1]["macro_auroc"] == 0.2


def test_best_by_selects_the_highest_metric() -> None:
    results = [
        ({"a": 1}, {"macro_auroc": 0.5}),
        ({"a": 2}, {"macro_auroc": 0.71}),
        ({"a": 3}, {"macro_auroc": 0.6}),
    ]
    cfg, report = best_by(results, "macro_auroc")
    assert cfg == {"a": 2}
    assert report["macro_auroc"] == 0.71
