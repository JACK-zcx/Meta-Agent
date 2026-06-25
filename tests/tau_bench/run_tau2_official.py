import sys, os, json, io
from datetime import datetime
for p in [r"F:\desktop\Meta-Agent-with-More-Agents-main\tau2-bench-main\src",
          r"F:\desktop\tau2-bench-main\src",
          r"F:\desktop\Meta-Agent-with-More-Agents-main"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

import tau2.utils.llm_utils as llm_utils
def _safe_cost(*args, **kwargs):
    return 0.0
llm_utils.get_response_cost = _safe_cost
"""τ2-bench official — MetaAgent adapter"""
os.chdir(r"F:\desktop\Meta-Agent-with-More-Agents-main")
os.environ["TAU2_DATA_DIR"] = r"F:\desktop\tau2_result"

from dotenv import load_dotenv; load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ.get("LLM_API_KEY", "")
os.environ["OPENAI_API_BASE"] = os.environ.get("LLM_BASE_URL", "").rstrip("/")
os.environ["TERM"] = "dumb"
os.environ["NO_COLOR"] = "1"
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Monkey-patch: patch the SOURCE FILE before tau2 imports ──
# Read user_simulator.py, find "from tau2.utils.llm_utils import generate", wrap it
_US_SRC = r"F:\desktop\Meta-Agent-with-More-Agents-main\tau2-bench-main\src\tau2\user\user_simulator.py"
with open(_US_SRC, "r", encoding="utf-8") as f:
    _us_original = f.read()

# Replace the LiteLLM generate call with a marker we can swap at runtime
if "OUR_GENERATE_PATCHED" not in _us_original:
    # Add a shim: at module level, import our function
    _new_content = _us_original.replace(
        "from tau2.utils.llm_utils import generate",
        "from tau2.utils.llm_utils import generate  # original\n"
        "import sys, os as _os\n"
        "def _patched_generate(*args, **kwargs):\n"
        "    from tests.tau_bench.adapter import _generate_for_user_simulator\n"
        "    return _generate_for_user_simulator(*args, **kwargs)\n"
        "generate = _patched_generate  # OUR_GENERATE_PATCHED\n",
    )
    with open(_US_SRC, "w", encoding="utf-8") as f:
        f.write(_new_content)
    print("[runner] Patched user_simulator.py source file")

# ── Register ──
from tests.tau_bench.adapter import register_meta_agent
register_meta_agent()

# ── Run ──
DOMAIN = os.environ.get("TAU2_DOMAIN", "mock")
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(result_dir, exist_ok=True)
save_path = os.path.join(result_dir, f"{DOMAIN}_{run_id}")


'mock', 'airline', 'retail', 'telecom', 'telecom-workflow', 'banking_knowledge'
sys.argv = [
    "tau2", "run",
    "--domain", 'airline',
    "--agent", "meta_agent",
    "--user", "user_simulator",
    "--agent-llm", f"openai/{os.environ.get('LLM_MODEL_ID')}",
    "--user-llm", f"openai/{os.environ.get('LLM_MODEL_ID')}",
    # "--num-tasks", "all",
    "--seed", "42",
    "--save-to", "tau2_result"
]

from tau2.cli import main
main()
