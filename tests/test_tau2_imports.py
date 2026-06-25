import sys
# 'mock', 'airline', 'retail', 'telecom', 'telecom-workflow', 'banking_knowledge'
sys.argv = [
    "tau2",
    "run",
    "--domain", "airline",
    "--agent", "meta_agent",

    "--agent-llm", "deepseek-v4-flash",
    "--user-llm", "deepseek-v4-flash",

    "--num-tasks", "5",
]

sys.path.insert(
    0,
    r"F:\desktop\Meta-Agent-with-More-Agents-main"
)

from tests.tau_bench.adapter import register_meta_agent
register_meta_agent()

from tau2.cli import main

if __name__ == "__main__":
    main()