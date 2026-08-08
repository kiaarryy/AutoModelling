from autofmu.metrics import regression_metrics


def test_perfect_fit():
    m = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["N"] == 3
    assert m["RMSE"] == 0.0
    assert m["MAPE_pct"] == 0.0


def test_known_error():
    m = regression_metrics([10.0, 10.0], [11.0, 9.0])
    assert round(m["RMSE"], 6) == 1.0
    assert round(m["MAPE_pct"], 6) == 10.0


def test_raw_interval_metrics_are_not_labelled_gl14():
    m = regression_metrics([10.0, 10.0], [11.0, 9.0])
    assert m["criterion"] == "raw_interval_custom"
    assert "GL14_pass" not in m
