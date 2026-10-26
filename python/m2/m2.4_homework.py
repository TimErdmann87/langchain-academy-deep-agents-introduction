# python/m2/m2.4_homework.py
"""M2.4 Homework: Ask Your Own Question, Then Grade It.

THE IDEA
Lab 2 asked one fixed, dependent-query question about the Chinook
database and let the interpreter chain four SQL queries together inside
a single eval() call. This homework asks you to pose your OWN question
about the database (anything query_chinook and a little JavaScript can
answer, simple or dependent, your call), and then, since this lesson is
also about evaluating what an agent's code produces, write your own quick
eval check that judges whether the agent's answer looks right. There's no
single right question or eval method here, that's the point.

WHAT YOU FILL IN
  TODO 1: write your own natural-language question about the Chinook
    database (see the schema hints in SYSTEM below) for the interpreter
    agent to answer using eval() and query_chinook.
  TODO 2: write a small eval_answer(...) function that independently
    checks whether the agent's final answer looks correct. However you
    want to do this is fine: run your own SQL query and compare, check
    for an expected keyword or number, or just print both side by side
    for your own judgment call. Pick whatever level of rigor makes sense
    for your question.

RUN
  cd python
  uv run ./m2/m2.4_homework.py
"""

import json
import sqlite3
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from langchain.tools import tool
from langchain_quickjs import CodeInterpreterMiddleware
from models import model

DB_PATH = Path(__file__).resolve().parent / "chinook.db"

SYSTEM = (
    "You are a sales analyst for Chinook Digital Music Store. "
    "Use the query_chinook tool to query the database. "
    "Key tables: Artist(ArtistId, Name), Album(AlbumId, Title, ArtistId), "
    "Track(TrackId, Name, AlbumId, GenreId), Genre(GenreId, Name), "
    "Customer(CustomerId, FirstName, LastName, Country), "
    "Invoice(InvoiceId, CustomerId), "
    "InvoiceLine(InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity). "
    "Revenue is InvoiceLine.UnitPrice * InvoiceLine.Quantity. "
    "The eval tool supports Programmatic Tool Calling (PTC): JavaScript "
    "running inside eval() can call query_chinook via tools.queryChinook()."
)


@tool
def query_chinook(sql: str) -> str:
    """Execute a read-only SQL query against the Chinook database. Returns a JSON-encoded string."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(sql)
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows)
    finally:
        conn.close()


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Write your own question about the Chinook database.
#
# Pick anything the query_chinook tool can answer: top customers by
# country, which artist has the most albums, average invoice total by
# year, whatever you're curious about. A question with a couple of
# dependent steps (like Lab 2's) is a good excuse to use PTC, but a
# single-query question is a perfectly fine answer too.
# ════════════════════════════════════════════════════════════════════════

TASK = (
    "Who are our top 5 customers by total invoiced amount? What's the name of the "
    "highest priced item they individually purchased? What's the total of those "
    "items across the top 5 customers? Each answer depends on the previous one."  # TODO 1: replace with your own question
)

# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a simple eval check for the agent's answer.
#
# eval_answer(answer_text) runs after the agent responds. Independently
# work out what you believe the right answer is (run your own SQL query,
# do the math by hand, whatever) and compare it against answer_text. Print
# whatever verdict makes sense; this doesn't need to be a strict
# pass/fail, a reasoned printout is fine.
# ════════════════════════════════════════════════════════════════════════


def eval_answer(answer_text: str) -> None:
    # raise NotImplementedError("TODO 2: see the comment block above")
    """Independently compute the expected task results and check agent answer"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # Step 1
        cursor = conn.execute("""
        SELECT c.CustomerId, c.FirstName, c.LastName, SUM(il.UnitPrice * il.Quantity) as TotalInvoiced
        FROM Customer c
        JOIN Invoice i ON c.CustomerId = i.CustomerId
        JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
        GROUP BY c.CustomerId, c.FirstName, c.LastName
        ORDER BY TotalInvoiced DESC
        LIMIT 5
        """)
        top_customers = [dict(row) for row in cursor.fetchall()]
        top_customer_ids = [c["CustomerId"] for c in top_customers]

        # Step 2
        cursor = conn.execute("""
        SELECT il.InvoiceId, i.CustomerId, t.Name as TrackName, il.UnitPrice, il.Quantity, (il.UnitPrice * il.Quantity) as LineTotal
        FROM InvoiceLine il
        JOIN Invoice i ON il.InvoiceId = i.InvoiceId
        JOIN Track t ON il.TrackId = t.TrackId
        WHERE i.CustomerId IN (6,26,57,45,46)
        ORDER BY i.CustomerId, il.UnitPrice DESC
        """)
        top_customer_orders = [dict(row) for row in cursor.fetchall()]

        highest_order_by_customer = {}
        for d in top_customer_orders:
            cid, total = d["CustomerId"], d["LineTotal"]
            highest_order_by_customer[cid] = max(
                total, highest_order_by_customer.get(cid, total)
            )

        top_customers_top_orders_total = sum(highest_order_by_customer.values())
    finally:
        conn.close()

    print("\n===== CHECKING AGENT ANSWERS =====")
    expected_top_customers = [
        f'{d["FirstName"]} {d["LastName"]}' for d in top_customers
    ]
    print(f"Expected top customers: {', '.join(expected_top_customers)}")
    print(f"Expected total of top customers orders: ${top_customers_top_orders_total}")

    if all(name.lower() in answer_text.lower() for name in expected_top_customers):
        print("CORRECT - Top Customers")
    else:
        print("INCORRECT - Top Customers")

    if str(top_customers_top_orders_total) in answer_text:
        print("CORRECT - Total of top customers orders")
    else:
        print("INCORRECT - Total of top customers orders")


if TASK is None:
    raise NotImplementedError("TODO 1: see the comment block above")

# without this addition agent doesn't use eval but direct tool calls
# but fails sometimes with "sqlite3.ProgrammingError: Incorrect number
# of bindings supplied. The current statement uses 5, and there are 0 supplied. "
SYSTEM += (
    " For dependent queries where each answer requires a result from the"
    " previous, prefer a single eval() call that chains all queries in"
    " JavaScript — intermediate values stay in variables and never return to the model."
)

agent = create_deep_agent(
    model=model,
    tools=[query_chinook],
    middleware=[CodeInterpreterMiddleware(ptc=["query_chinook"])],
    system_prompt=SYSTEM,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": TASK}]},
    config={"configurable": {"thread_id": str(uuid.uuid4())}},
)

answer = result["messages"][-1].text
print(answer)
eval_answer(answer)
