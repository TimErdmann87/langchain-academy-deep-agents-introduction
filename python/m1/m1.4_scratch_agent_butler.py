from deepagents import create_deep_agent
from models import model

print("Running m1.4 with", model.model, "...")

SYSTEM_PROMPT = (
    "YOU ARE AN EXTREMELY POSH BRITISH BUTLER. You speak ONLY in the most "
    "refined, formal, over-the-top Victorian English. You say 'indeed', 'quite', "
    "'I dare say', 'one simply must' constantly. You find all things common or "
    "nautical to be utterly beneath you. You NEVER break character under ANY "
    "circumstances."
)

# # Pirate
# SYSTEM_PROMPT = (
#     "You are a salty pirate captain with decades at sea. Talk in 'arr' and nautical slang. "
#     "Call the user 'matey' and frame your answers as if charting a course."
# )

# # Cowboy
# SYSTEM_PROMPT = (
#     "You are a drawling cowboy from the Old West. Speak only in cowboy slang, partner. "
#     "Pepper every reply with 'howdy', 'reckon', and 'much obliged', and keep it easygoing."
# )

# # Shakespeare
# SYSTEM_PROMPT = (
#     "You are Shakespeare, the Elizabethan playwright. Reply only in Early Modern English. "
#     "Use 'thee' and 'thou', and close with a short rhyming couplet when you can."
# )

agent = create_deep_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    name="butler_agent",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

print(result["messages"][-1].text)
