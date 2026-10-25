from time import perf_counter

from deepagents import create_deep_agent
from models import model

t0: float = perf_counter()
print("Running m1.2 with", model.model, "...")

agent = create_deep_agent(model=model)

result = agent.invoke({"messages": [{"role": "user", "content": "What is an LLM?"}]})

t: float = perf_counter() - t0
print(f"\nTime: {int(t//60)} min {round(t % 60)} s")
print("\nAgent:")
print(result["messages"][-1].text)
print()
