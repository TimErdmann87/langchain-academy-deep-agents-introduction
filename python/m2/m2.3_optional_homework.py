"""Module 2: virtual_mode scopes the file tools, never execute.

LocalShellBackend runs commands directly on THIS machine with no isolation.
Every `execute` call is gated behind your approval. Only approve a command
you would be comfortable typing into a terminal yourself.

Run from the python/ folder:  uv run python .\\m2\\m2_local_shell_escape.py
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from models import model

# --- Fresh scratch directory: the only place the FILE tools can reach ---------
scratch_dir = Path(__file__).parent / "sandbox_scratch"
scratch_dir.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant on a Windows machine. "
    "The execute tool runs commands in the Windows command shell (cmd.exe), "
    "so use Windows commands such as `cd` and `dir`, not bash commands like `ls`."
)

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    backend=LocalShellBackend(root_dir=str(scratch_dir), virtual_mode=True),
    interrupt_on={"execute": {"allowed_decisions": ["approve", "reject"]}},
    checkpointer=MemorySaver(),
)

DEFAULT_REJECTION = (
    "The user rejected this command. Do not try to run it another way; "
    "tell the user what you would have done instead."
)


def review(action_requests: list[dict]) -> list[dict]:
    """Show each pending execute call and collect an approve/reject decision."""
    decisions: list[dict] = []
    for req in action_requests:
        print("\n" + "=" * 60)
        print(f"PENDING TOOL CALL: {req['name']}")
        for key, value in req["args"].items():
            print(f"  {key}: {value}")
        print("=" * 60)
        while True:
            choice = input("Approve or reject? [a/r]: ").strip().lower()
            if choice in ("a", "approve"):
                decisions.append({"type": "approve"})
                break
            if choice in ("r", "reject"):
                msg = input("Rejection message (Enter for default): ").strip()
                decisions.append(
                    {"type": "reject", "message": msg or DEFAULT_REJECTION}
                )
                break
    return decisions


def run(task: str, thread_id: str) -> None:
    """Run one task, pausing for review whenever execute is called."""
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n\n##### TASK ({thread_id}) #####\n{task}")

    result = agent.invoke({"messages": [{"role": "user", "content": task}]}, config)
    while result.get("__interrupt__"):
        interrupt = result["__interrupt__"][0]
        decisions = review(interrupt.value["action_requests"])
        result = agent.invoke(Command(resume={"decisions": decisions}), config)

    # What execute actually returned (real output, or your rejection message)
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage) and msg.name == "execute":
            print("\n--- execute tool result ---")
            print(str(msg.content)[:1500])

    print("\n--- Agent's final answer ---")
    print(result["messages"][-1].text)


# --- Task 1: write_file, ungated, confined to root_dir --------------------------
TASK_WRITE = (
    "Use write_file to create /notes/hello.txt containing the text "
    "'Hello from the scratch directory'. Then list /notes to confirm it exists."
)

# --- Task 2: execute on a read-only target OUTSIDE root_dir --------------------
# Approve this one after checking the command is only a directory listing.
TASK_ESCAPE = (
    f"Use the execute tool to list the contents of my home directory, {Path.home()}. "
    "Show me the output."
)

# --- Task 3: practice rejecting -------------------------------------------------
# Reject this one and see how the agent responds to your message.
TASK_REJECT = "Use the execute tool to print the current date and time."


if __name__ == "__main__":
    run(TASK_WRITE, "local-shell-write")
    print("\nFiles on disk under sandbox_scratch:")
    for path in sorted(scratch_dir.rglob("*")):
        print("  ", path.relative_to(scratch_dir))

    run(TASK_ESCAPE, "local-shell-escape")
    run(TASK_REJECT, "local-shell-reject")
