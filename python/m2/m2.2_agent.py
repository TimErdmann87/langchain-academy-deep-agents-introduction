# python/m2/m2.2_agent.py
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from models import model

reference_dir = Path(__file__).parent / "reference"
reference_dir.mkdir(exist_ok=True)
(reference_dir / "chinook-sales.md").write_text("""\
# Chinook Sales Reference

You are a sales representative for Chinook Digital Music Store.

Responsibilities:
- Look up customer accounts and purchase history
- Recommend music based on genre and artist preferences
- Answer questions about artists, albums, tracks, and invoices
""")

agent = create_deep_agent(
    model=model,
    checkpointer=MemorySaver(),
    backend=CompositeBackend(
        default=StateBackend(),
        routes={
            "/reference/": FilesystemBackend(
                root_dir=str(reference_dir),
                virtual_mode=True,
            ),
        },
    ),
    permissions=[
        FilesystemPermission(
            operations=["write"],
            paths=["/reference/**"],
            mode="deny",
        ),
    ],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                # "content": (
                #     "Read /reference/chinook-sales.md, then add this note to it: "
                #     "'Current promotion: 20% off all Jazz albums through end of month.'"
                # ),
                # "content": (
                #     "Read /reference/chinook-sales.md, then create a new file under "
                #     "/new/chinook-sales-update.md and add the content of the referenced file plus this note: "
                #     "'Current promotion: 20% off all Jazz albums through end of month.'"
                # ),
                # "content": ("What files do you have access to?"),
                # "content": (
                #     "What about a folder called '/new'? If so, are there any files and in that case what's the content of the files?"
                # ),
                "content": (
                    "What did I just ask you?"
                ),  # MemorySaver as checkpointer was not set yet. And even if set, every run of the file is a new process with blank memory.
            }
        ]
    },
    config={"configurable": {"thread_id": "lab-m2.2"}},
)

# print(result["messages"][-1].text)


# Conversation within same process
messages: list[str] = [
    "Read /reference/chinook-sales.md, then add this note to it: "
    "'Current promotion: 20% off all Jazz albums through end of month.'",
    "Read /reference/chinook-sales.md, then create a new file under "
    "/new/chinook-sales-update.md and add the content of the referenced file plus this note: "
    "'Current promotion: 20% off all Jazz albums through end of month.'",
    "What files do you have access to?",
    "What about a folder called '/new'? If so, are there any files and in that case what's the content of the files?",
]

for msg in messages:
    result = agent.invoke(
        input={"messages": [HumanMessage(msg)]},
        config={"configurable": {"thread_id": "lab-m2.2"}},
    )
    agent_msg: str = result["messages"][-1].text

    print()
    print(10 * "- ")
    print()
    print("User:", msg)
    print("Agent:", agent_msg)
    print()
