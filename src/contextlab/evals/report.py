"""Report writing and formatting."""
import json
from datetime import datetime, timezone
from pathlib import Path

from contextlab.evals.types import CaseResult, CheckResult, EvalReport, SuiteMetrics


def format_summary(report: EvalReport) -> str:
    """Format a one-screen summary of the report."""
    lines = []
    lines.append("=== Eval Summary ===")
    for name, sm in report.suites.items():
        lines.append(f"  {name}: {sm.n_pass}/{sm.n} passed", )
        if sm.metrics:
            for k, v in sm.metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.3f}")
                else:
                    lines.append(f"    {k}: {v}")
    if report.gates:
        lines.append("Gates:")
        for g in report.gates:
            tag = "PASS" if g.passed else "FAIL"
            lines.append(f"  [{tag}] {g.name}: {g.detail}")
    worst = [c for c in report.cases if not c.passed][:5]
    if worst:
        lines.append("Worst failures:")
        for c in worst:
            lines.append(f"  {c.id} ({c.suite})")
    return "\n".join(lines)


def write_report(report: EvalReport, path: str | Path) -> None:
    """Write EvalReport to JSON file and set passed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2)


def write_latest(report: EvalReport) -> None:
    """Write as artifacts/eval_report.json (stable name)."""
    write_report(report, "artifacts/eval_report.json")


def write_timestamped(report: EvalReport) -> None:
    """Write with UTC timestamp filename."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    write_report(report, f"artifacts/eval_{ts}.json")
