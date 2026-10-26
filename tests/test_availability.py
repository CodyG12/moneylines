import pandas as pd
import pytest

from nbamodel.availability import availability_features
from nbamodel.data import DataError


def _rows():
    return pd.DataFrame({"date": pd.to_datetime(["2024-01-05"]), "home_team": ["A"], "away_team": ["B"],
                         "prediction_timestamp": pd.to_datetime(["2024-01-05T23:00Z"])})


def test_only_reports_published_before_prediction_time_count():
    reps = pd.DataFrame({
        "report_ts": ["2024-01-05T18:00Z", "2024-01-05T18:00Z", "2024-01-05T23:30Z", "2024-01-05T17:00Z",
                      "2024-01-05T20:00Z"],
        "date": ["2024-01-05"] * 5, "team": ["A", "A", "A", "B", "B"],
        "player": ["p1", "p2", "p3", "q1", "q1"],
        "status": ["out", "questionable", "out", "out", "available"], "impact": [0.3, 0.2, 0.5, 0.4, 0.4]})
    f = availability_features(_rows(), reps)
    assert f.loc[0, "home_n_out"] == 1 and f.loc[0, "home_impact_out"] == pytest.approx(0.3)  # p3 report too late
    assert f.loc[0, "away_n_out"] == 0                                    # q1's later (pre-deadline) report: available
    assert f.loc[0, "impact_out_diff"] == pytest.approx(0.3)


def test_reports_without_timestamps_rejected():
    reps = pd.DataFrame({"report_ts": [None], "date": ["2024-01-05"], "team": ["A"], "player": ["p"], "status": ["out"]})
    with pytest.raises(DataError, match="report_ts"):
        availability_features(_rows(), reps)
