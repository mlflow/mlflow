"""
This is an example for leveraging MLflow's auto tracing capabilities for Pydantic AI.
Most codes are from https://ai.pydantic.dev/examples/bank-support/.
"""

import mlflow
import mlflow.pydantic_ai

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Pydantic AI Example")
mlflow.pydantic_ai.autolog(disable=False)

from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext


class DatabaseConn:
    """This is a fake database for example purposes.

    In reality, you'd be connecting to an external database
    (e.g. PostgreSQL) to get information about customers.
    """

    @classmethod
    async def customer_name(cls, *, id: int) -> str | None:
        if id == 123:
            return "John"

    @classmethod
    async def customer_balance(cls, *, id: int, include_pending: bool) -> float:
        if id == 123 and include_pending:
            return 123.45
        else:
            raise ValueError("Customer not found")


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str = Field(description="Advice returned to the customer")
    block_card: bool = Field(description="Whether to block their card or not")
    risk: int = Field(description="Risk level of query", ge=0, le=10)


support_agent = Agent(
    "openai:gpt-4o",
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    system_prompt=(
        "You are a support agent in our bank, give the "
        "customer support and judge the risk level of their query. "
        "Reply using the customer's name."
    ),
    instrument=True,
)


@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(ctx: RunContext[SupportDependencies], include_pending: bool) -> str:
    """Returns the customer's current account balance."""
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f"${balance:.2f}"


if __name__ == "__main__":
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = support_agent.run_sync("What is my balance?", deps=deps)
    print(result.output)

    result = support_agent.run_sync("I just lost my card!", deps=deps)
    print(result.output)
