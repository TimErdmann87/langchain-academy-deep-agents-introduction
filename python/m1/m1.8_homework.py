# python/m1/m1.8_homework.py
"""M1.8 Homework: Gate Your Own Action Tool.

THE IDEA
Lab 1 gated one action tool, send_email, behind interrupt_on and walked
through approve/edit/reject on it. This homework asks you to do the same
thing for an action tool of your own choosing: post a tweet, book a
meeting room, place an order, delete a file, whatever you like.

WHAT YOU FILL IN
  TODO 1: define your own @tool-decorated action tool. Pick any
    side-effecting action you like; the function body can just return a
    confirmation string, the same way Lab 1's send_email did.
  TODO 2: configure interrupt_on for your tool with an allowed_decisions
    list of your choosing, and write a system prompt plus an initial user
    request that would lead the model to propose calling it.

The review loop below (borrowed from Lab 1, unchanged) prints any
pending tool call and asks you to approve, edit, or reject it. Run the
script more than once, picking a different choice each time, to see both
an approve/edit path and a reject path.

RUN
  cd python
  uv run ./m1/m1.8_homework.py
"""

import warnings

from langchain.agents.middleware import InterruptOnConfig, ToolCallRequest
from langchain_core.runnables import RunnableConfig

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from models import model

# ════════════════════════════════════════════════════════════════════════
# TODO 1: Define your own action tool.
#
# Requirements:
#   - Keep the @tool decorator.
#   - Give it a real docstring describing the action it performs.
#   - Have it take at least one argument and return a confirmation
#     string, the same way send_email returned a confirmation string
#     instead of actually sending mail.
#
# Example shape (delete this and write your own):
#   @tool
#   def post_tweet(content: str) -> str:
#       """Post a tweet with the given content."""
#       return f"Tweet posted: {content!r}"
# ════════════════════════════════════════════════════════════════════════


@tool
def pay_credit_card(card_name: str, amount: float) -> str:
    """Use this tool to pay my credit card by a certain amount"""
    return f"Payed ${amount} to {card_name}"


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Configure interrupt_on for your tool, and write a system prompt
# plus an initial user request that would lead the model to propose
# calling it.
#
# Requirements:
#   - interrupt_on should name your tool (rename your_action_tool if you
#     like) and an allowed_decisions list, e.g.
#     {"your_action_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
#   - SYSTEM_PROMPT should tell the agent when to use your tool.
#   - INITIAL_REQUEST should be a user message that would make the agent
#     want to call it.
# ════════════════════════════════════════════════════════════════════════


# Optional conditional interrupt
APPROVAL_THRESHOLD = 20


def amount_too_high(request: ToolCallRequest) -> bool:
    """Credit card payment too high and needs human inervention"""
    try:
        amount = float(request.tool_call["args"]["amount"])
    except (TypeError, ValueError, KeyError):
        return True  # missing or unparseable amount → interrupt
    return amount > APPROVAL_THRESHOLD


SYSTEM_PROMPT = f"You are an assistant helping with personal finance. You have access to the tool 'pay_credit_card' to assist the user paying a certain credit card. If the amount is higher than ${APPROVAL_THRESHOLD} the user will be asked to check the transaction before execution."
INITIAL_REQUEST = "I want to pay off my AMEX Gold credit card with $5."
INTERUPT_ON_CONFIG: InterruptOnConfig = {
    "allowed_decisions": ["approve", "edit", "reject"],
    "when": amount_too_high,  # Optional conditional interrupt
}
INTERRUPT_ON: dict[str, bool | InterruptOnConfig] = {
    "pay_credit_card": INTERUPT_ON_CONFIG
}

# Guards against running with an unfilled placeholder; the filled
# reference doesn't need this since there's no placeholder text left.
if "TODO 1" in pay_credit_card.description:
    raise NotImplementedError("TODO 1: see the comment block above")
if "TODO 2" in SYSTEM_PROMPT or "TODO 2" in INITIAL_REQUEST:
    raise NotImplementedError("TODO 2: see the comment block above")

agent = create_deep_agent(
    model=model,
    tools=[pay_credit_card],
    system_prompt=SYSTEM_PROMPT,
    interrupt_on=INTERRUPT_ON,
    checkpointer=MemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "m1-8-homework-demo"}}

result = agent.invoke(
    {"messages": [{"role": "user", "content": INITIAL_REQUEST}]},
    config=config,
    version="v2",
)

while result.interrupts:
    pending = result.interrupts[0].value
    decisions = []
    for req in pending["action_requests"]:
        print(f"\nApproval required for {req['name']}:")
        print(req["args"])

        choice = (
            input("\nApprove, edit, or reject? (approve/edit/reject): ").strip().lower()
        )
        if choice in ("approve", "yes", "y"):
            decisions.append({"type": "approve"})
        elif choice in ("edit", "e"):
            edited_args = dict(req["args"])
            for key, curr in edited_args.items():
                edited_args[key] = input(f"New value for '{key}' (current: {curr}): ")
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {"name": req["name"], "args": edited_args},
                }
            )
        else:
            decisions.append(
                {"type": "reject", "message": "User rejected this action."}
            )

    result = agent.invoke(
        Command(resume={"decisions": decisions}), config=config, version="v2"
    )

for msg in result.value["messages"]:
    if hasattr(msg, "name") and msg.name == "pay_credit_card":
        print(msg.text)
        break
