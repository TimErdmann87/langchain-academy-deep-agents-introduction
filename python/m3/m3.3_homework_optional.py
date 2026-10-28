"""Shared team workspace memory.

The homework scoped memory to ("memory", workspace_id, user_id), which gives
every user a private notebook. This script drops user_id from the namespace
so every member of a workspace reads and writes the same AGENTS.md, like a
shared team knowledge base.

Flow:
  1. Seed a team AGENTS.md under ("memory", "acme").
  2. alice (workspace "acme") tells the agent to remember a new team fact.
  3. bob (same workspace, different user) asks about that fact in a fresh run.
     He shares no message history with alice, only the store.
  4. carol (workspace "globex") asks the same question as a control: she
     should NOT see acme's memory.
"""

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.store.memory import InMemoryStore
from models import model

store = InMemoryStore()

MEMORY_PATH = "/memories/AGENTS.md"  # path the agent sees
STORE_KEY = "/AGENTS.md"  # key in the store (CompositeBackend strips "/memories")

alice_context = {"workspace_id": "acme", "user_id": "alice"}
bob_context = {"workspace_id": "acme", "user_id": "bob"}
carol_context = {"workspace_id": "globex", "user_id": "carol"}

# A distinctive fact the model cannot guess, so bob's answer proves the share.
TEAM_FACT_MARKER = "ship.sh"


def namespace_for_workspace(context: dict) -> tuple[str, ...]:
    """Workspace-scoped namespace. user_id is left out on purpose."""
    return ("memory", context["workspace_id"])


def workspace_namespace(runtime) -> tuple[str, ...]:
    """Namespace callable for StoreBackend; reads the per-invoke context."""
    return namespace_for_workspace(runtime.context)


def read_team_memory(context: dict) -> str | None:
    """Read AGENTS.md for a workspace straight from the store."""
    item = store.get(namespace_for_workspace(context), STORE_KEY)
    return item.value["content"] if item else None


# 1. Seed the shared team knowledge base for workspace "acme".
store.put(
    namespace_for_workspace(alice_context),
    STORE_KEY,
    create_file_data("""\
# Acme Team Knowledge Base
 
Shared by everyone in the acme workspace. Add facts the whole team needs.
 
## Conventions
- Python 3.12, managed with `uv`
- Run tests with: `uv run pytest`
"""),
)

agent = create_deep_agent(
    model=model,
    name="Team_Memory_Agent",
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=workspace_namespace)},
    ),
    store=store,
    memory=[MEMORY_PATH],
    system_prompt=(
        "You are a coding assistant for a software team. "
        f"{MEMORY_PATH} is the team's shared knowledge base: every teammate "
        "in this workspace reads it. When someone asks you to remember "
        "something for the team, add it to that file."
    ),
)


def ask(context: dict, question: str) -> str:
    """Run one fresh agent invocation (no shared thread) under a context."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        context=context,
    )
    return result["messages"][-1].text


# 2. alice teaches the agent a team fact.
print(f"=== alice ({alice_context}) ===")
print(
    ask(
        alice_context,
        "Remember this for the whole team: we deploy to staging with "
        "`./scripts/ship.sh --env staging`, and only after CI is green. "
        "Update the team memory.",
    )
)

acme_memory = read_team_memory(alice_context)
print("\n--- acme AGENTS.md after alice's write ---")
print(acme_memory)

# 3. bob asks in his own context. Same workspace, different user, new run.
QUESTION = "How do we deploy to staging?"

print(f"\n=== bob ({bob_context}) ===")
bob_answer = ask(bob_context, QUESTION)
print(bob_answer)

# 4. Control: a different workspace should not see acme's memory.
print(f"\n=== carol ({carol_context}) ===")
carol_answer = ask(carol_context, QUESTION)
print(carol_answer)

# Verify with the store and the answer text, not tool calls: memory=[...]
# injects AGENTS.md into the system prompt, so bob never needs read_file.
print("\n=== Checks ===")
# List items, not namespaces: InMemoryStore registers an empty namespace on
# any read, so list_namespaces() would also show "globex" after carol's run.
print("Files in store (namespace, key):")
for item in store.search(("memory",)):
    print(f"  {item.namespace}  {item.key}")

checks = {
    "alice's fact is in acme's shared AGENTS.md": (
        acme_memory is not None and TEAM_FACT_MARKER in acme_memory
    ),
    "bob and alice resolve to the same namespace": (
        namespace_for_workspace(bob_context) == namespace_for_workspace(alice_context)
    ),
    "bob's answer uses alice's fact": TEAM_FACT_MARKER in bob_answer,
    "globex has no copy of acme's memory": read_team_memory(carol_context) is None,
    "carol's answer does not use acme's fact": TEAM_FACT_MARKER not in carol_answer,
}
for name, passed in checks.items():
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")
