# python/m1/m1.6_homework.py
"""M1.6 Homework: Connect to a Different MCP Server.

THE IDEA
Lab 1 connected to the LangChain docs MCP server and filtered its tools
down to just search_docs_by_lang_chain. This homework asks you to connect
to a different public MCP server entirely, one that requires no auth
beyond what your labs already use, and put one of its tools to work.

Don't know where to look? A few free, no-auth public servers to try:
  - DeepWiki (https://mcp.deepwiki.com/mcp): ask questions about any
    public GitHub repo's code and docs.
  - X Docs (https://docs.x.com/mcp): search and retrieve X's public API
    documentation.
Or find your own!

WHAT YOU FILL IN
  TODO 1: build and return the filtered list of MCP tools to use from a
    server and filter of your own choosing.
  TODO 2: write a question suited to your chosen server's own domain,
    not Lab 1's "what is MCP..." question, which won't make sense to
    ask a server about GitHub repos, API docs, or whatever you picked.

RUN
  cd python
  uv run ./m1/m1.6_homework.py
"""

import asyncio
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from models import model

# ════════════════════════════════════════════════════════════════════════
# TODO 1: Build your MCP client and return its filtered tool list.
#
# Requirements:
#   - Point "url" at a different public MCP server than Lab 1's
#     docs-langchain server (no auth/API key required beyond what your
#     labs already use). See the module docstring for two ready-to-use
#     options.
#   - Fetch tools with client.get_tools() and filter them the same way
#     Lab 1 did, with an ALLOWED set.
#
# Example shape (delete this and write your own):
#   async def build_tools():
#       client = MultiServerMCPClient({
#           "my-server": {"transport": "http", "url": "https://..."}
#       })
#       tools = await client.get_tools()
#       ALLOWED = {"some_tool_name"}
#       return [t for t in tools if t.name in ALLOWED]
# ════════════════════════════════════════════════════════════════════════


async def build_tools():
    client = MultiServerMCPClient(
        {"deep-wiki": {"transport": "http", "url": "https://mcp.deepwiki.com/mcp"}}
    )
    tools = await client.get_tools()
    for t in tools:
        print()
        print(t.name)
        print(t.description[:100])

    BLOCKED_TOOLS = {"read_wiki_structure"}

    tools_final = [t for t in tools if t.name not in BLOCKED_TOOLS]
    print("\nSelected Tools:", f"{', '.join([t.name for t in tools_final])}\n")

    return tools_final


# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a question suited to your chosen server's own domain,
# not Lab 1's "what is MCP..." question.
# ════════════════════════════════════════════════════════════════════════

# QUESTION = "What can you help me with?"
QUESTION = "Using DeepWiki, explain how the langchain-ai/deepagents repo implements subagents. First look at the wiki contents to find the relevant section, then ask a question about how the main agent hands a task to a subagent."


async def main():
    tools = await build_tools()
    agent = create_deep_agent(model=model, tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
    print()
    print("User:", QUESTION)
    print("Agent:", result["messages"][-1].text)
    print()


asyncio.run(main())
