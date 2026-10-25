# OPTIONAL
# What you'll build: two divergent futures branched from the same past checkpoint.

# Requirements:

# Write your own create_deep_agent(model=model, checkpointer=MemorySaver()) call from scratch, in a new file under python/m1/.
# On a single thread, send a few turns, then call agent.get_state_history(thread) and pick an older snapshot (not the latest).
# Resume from that snapshot's config twice, sending a different message each time, so you get two branches that share the same earlier history but diverge from that point on.
# Call get_state_history again and confirm you can see checkpoints from both branches tracing back to the shared snapshot you picked.


from deepagents import create_deep_agent
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from models import model

agent = create_deep_agent(model=model, checkpointer=MemorySaver())

thread: RunnableConfig = {"configurable": {"thread_id": "t_id_01"}}

user_messages: list[str] = [
    "How can I increase my productivity? Please list 5 different ways that I can implement immediately.",
    "Thank you, which one do you think offers the best productivity enhancements?",
    "Is silence better than certain acustic 'flow state' background music?",
]

for msg in user_messages:
    res = agent.invoke(input={"messages": [HumanMessage(content=msg)]}, config=thread)

    print("\nUser:\t", msg)
    print("Agent:\t", res["messages"][-1].content)

print()
print("= = = = = = = = ANALYZING STATE HISTORY = = = = = = = =")
print()

state_history = list(agent.get_state_history(thread))
print(len(state_history))
print(state_history[0])
