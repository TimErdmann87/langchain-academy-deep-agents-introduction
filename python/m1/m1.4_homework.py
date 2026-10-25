# python/m1/m1.4_homework.py
"""M1.4 Homework: Scope the Agent to One Domain.

THE IDEA
Lab 1 had you swap personas (pirate, cowboy, Shakespeare) on top of the
butler system prompt, which only changes the agent's voice. This homework
uses `system_prompt` differently: instead of persona, write a constraint
that scopes the agent to a single domain of your choosing (cooking,
houseplants, retro video games, personal finance, etc.) and
instructs it to refuse or redirect anything outside that domain.

There's no single correct domain here, that's the point. What matters is
that the refusal actually holds, not just that the agent sounds like
something.

WHAT YOU FILL IN
  TODO 1: write your own SYSTEM_PROMPT string that scopes the agent to a
    single domain of your choosing and tells it to refuse or redirect
    anything outside that domain (no persona/voice requirement here,
    just the scope + refusal instruction).
  TODO 2: invoke the agent with two test prompts, one inside your domain
    and one clearly outside it, and print both responses so you can see
    whether the refusal actually held.

RUN
  cd python
  uv run ./m1/m1.4_homework.py
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from textwrap import dedent

from deepagents import create_deep_agent
from models import model

# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write a system prompt that scopes the agent to one domain and
# tells it to refuse or redirect anything outside that domain.
#
# Requirements:
#   - Pick one domain (a subject, not a persona).
#   - State clearly what the agent should do when asked about something
#     outside that domain (e.g. say it can't help, redirect back to the
#     domain, ask a domain-relevant follow-up).
#
# Example shape (delete this and write your own):
#   SYSTEM_PROMPT = (
#       "You only answer questions about ... . If asked about anything "
#       "else, ... ."
#   )
# ════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = dedent("""\
  You are a helpful and polite assistant for personal fitness. You only reply to the user about topics covering personal fitness like nutrition, bio mechanics, workout programming, exercises, recovery, and so on.
  
  Important:
  If any other topic than personal fitness gets raised by the user, you must politely explain in brief that you are not answering about other topics than personal fitness.
  
  Final Check:
  Before sending your response to the user, check your answer again and only send it to the user if your answer is about personal fitness.\
  """)  # noqa: E501,W293

# print(repr(SYSTEM_PROMPT)) # to check where \n are placed
print(SYSTEM_PROMPT)

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="Homework_Agent",
)


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Run one in-domain prompt and one out-of-domain prompt through
# the agent and print both responses, so you can check whether the
# refusal actually held.
# ════════════════════════════════════════════════════════════════════════


def interact_w_agent(msg: str) -> str:
    res = agent.invoke({"messages": [{"role": "user", "content": msg}]})
    return res["messages"][-1].text


def run_test_prompts():
    """TODO 2: invoke `agent` with one in-domain prompt and one
    out-of-domain prompt, and print each response."""

    msg1 = "How many sets and reps should my workout cover per muscle group. Let's say for a back and biceps day."
    res1 = interact_w_agent(msg1)
    print("\nUser: ", msg1)
    print("\nAgent: ", res1)
    print("\n- - - - -\n")

    msg2 = "What skills do I need to develop to be ready for the job market in 2026 when I want to get into AI and Agentic AI for drug discovery?"
    res2 = interact_w_agent(msg2)
    print("\nUser: ", msg2)
    print("\nAgent: ", res2)

    return


run_test_prompts()
