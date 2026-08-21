from dual_backend.cli import _compact_summary


def test_compact_summary_uses_actual_report_path() -> None:
    summary = {
        "dataset": {"rows": {"train": 10}},
        "backends": {"torch": {"metrics": {"rmse": 1.2}}},
        "comparison": {},
        "report_path": "custom_reports/run_summary.json",
    }

    compact = _compact_summary(summary)

    assert compact["report"] == "custom_reports/run_summary.json"
