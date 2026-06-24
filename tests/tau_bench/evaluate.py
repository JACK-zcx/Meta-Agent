"""
Evaluation runner — tests MetaAgent against τ-bench-style task suite.

Usage:
    python tests/tau_bench/evaluate.py           # run all tasks
    python tests/tau_bench/evaluate.py --report  # run + generate report
"""
import json
import os
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

# ── Path setup ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from server import app
from scorer import Scorer, score_task

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Server management ──
_server_thread = None


def start_server(port=8765):
    """Start FastAPI server in background thread."""
    global _server_thread

    def run():
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="error")

    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()
    time.sleep(4)

    # Verify
    import urllib.request
    for i in range(10):
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
            if r.status == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


# ── Task execution ──
def run_task_turn(query: str, timeout: int = 120) -> tuple[str, list[dict]]:
    """Send one query via WebSocket and collect all events until 'done'."""
    import websocket
    events = []
    answer = ""

    try:
        ws = websocket.create_connection("ws://127.0.0.1:8765/ws", timeout=5)
        ws.settimeout(timeout)
        ws.send(query)
        while True:
            msg = ws.recv()
            if not msg:
                break
            evt = json.loads(msg)
            events.append(evt)
            if evt.get("type") == "final_result":
                answer = evt.get("answer", "")
            if evt.get("type") == "done":
                break
    except Exception as e:
        events.append({"type": "_err", "msg": str(e)})
    finally:
        try:
            ws.close()
        except Exception:
            pass

    return answer, events


def run_all_tasks(tasks_file: str = None) -> list[dict]:
    """Run all tasks and return scored results."""
    if tasks_file is None:
        tasks_file = str(Path(__file__).resolve().parent / "tasks.json")

    with open(tasks_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    results = []
    print(f"Running {len(tasks)} tasks...\n")

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        category = task.get("category", "?")
        turns = task.get("turns", [])
        print(f"[{i + 1}/{len(tasks)}] {task_id} ({category}, {len(turns)} turns) ...", end=" ", flush=True)

        all_answers = []
        all_events = []

        for turn in turns:
            query = turn.get("query", "")
            answer, events = run_task_turn(query, timeout=120)
            all_answers.append(answer)
            all_events.append(events)

        result = score_task(task, all_events, all_answers)
        status = "PASS" if result.get("passed") else "FAIL"
        print(f"{status} (score={result['score']:.2f}, dispatch={result['dispatch_count']})")
        results.append(result)

    return results


# ── Report generation ──
def generate_report(results: list[dict], output_file: str = None) -> str:
    """Generate a markdown evaluation report."""
    if output_file is None:
        output_file = str(RESULTS_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    avg_score = sum(r["score"] for r in results) / max(total, 1)

    # By category
    categories = {}
    for r in results:
        cat = r.get("category", "other")
        if cat not in categories:
            categories[cat] = {"total": 0, "passed": 0, "score_sum": 0.0}
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1
        categories[cat]["score_sum"] += r["score"]

    lines = [
        "# Meta-Agent Evaluation Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n## Summary",
        f"\n| Metric | Value |",
        f"|--------|-------|",
        f"| Total Tasks | {total} |",
        f"| Passed | {passed} |",
        f"| Pass Rate | {passed / max(total, 1) * 100:.1f}% |",
        f"| Average Score | {avg_score:.3f} |",
        f"\n## By Category",
        f"\n| Category | Tasks | Passed | Rate | Avg Score |",
        f"|----------|-------|--------|------|-----------|",
    ]

    for cat, stats in sorted(categories.items()):
        rate = stats["passed"] / max(stats["total"], 1) * 100
        avg = stats["score_sum"] / max(stats["total"], 1)
        lines.append(f"| {cat} | {stats['total']} | {stats['passed']} | {rate:.0f}% | {avg:.3f} |")

    lines.append(f"\n## Detailed Results\n")
    lines.append("| # | Task | Category | Score | Dispatches | Errors | ToolAgent |")
    lines.append("|---|------|----------|-------|------------|--------|-----------|")
    for i, r in enumerate(results):
        ta = "Yes" if r.get("tool_agent_triggered") else ""
        lines.append(
            f"| {i + 1} | {r['task_id']} | {r.get('category', '?')} "
            f"| {r['score']:.2f} | {r['dispatch_count']} "
            f"| {r['error_count']} | {ta} |"
        )

    report = "\n".join(lines)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report)

    # Also save raw JSON results
    json_file = output_file.replace(".md", ".json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    return report


# ── CLI ──
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Meta-Agent evaluation")
    parser.add_argument("--report", action="store_true", help="Generate report after run")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    args = parser.parse_args()

    print("=" * 50)
    print("Meta-Agent Evaluation Suite (τ-bench format)")
    print("=" * 50)

    # Start server
    print("\nStarting evaluation server on port 8765...")
    if not start_server(args.port):
        print("ERROR: Server failed to start!")
        sys.exit(1)
    print("Server ready.\n")

    # Run tasks
    results = run_all_tasks()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = str(RESULTS_DIR / f"run_{timestamp}.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    print(f"\n{'=' * 50}")
    print(f"RESULTS: {passed}/{total} passed ({passed / max(total, 1) * 100:.1f}%)")
    print(f"Results saved to: {json_file}")

    if args.report:
        report = generate_report(results)
        print(report)
