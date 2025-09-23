from functools import partial
from typing import Callable
from src.graph.state import AgentState


def _extract_base_agent_key(unique_id: str) -> str:
    parts = unique_id.split('_')
    if len(parts) >= 2 and len(parts[-1]) == 6 and parts[-1].isalnum():
        return '_'.join(parts[:-1])
    return unique_id


def create_agent_function(agent_function: Callable, agent_id: str) -> Callable[[AgentState], dict]:
    """
    Creates a new function from an agent function that accepts an agent_id.

    Adds a lightweight gating layer: if state.data.active_agents is provided and the
    base agent key is not present, the wrapped function becomes a no-op for this step.

    :param agent_function: The agent function to wrap.
    :param agent_id: The ID to be passed to the agent.
    :return: A new function that can be called by LangGraph.
    """

    def maybe_skip(state: AgentState) -> dict:
        data = state.get("data", {})
        active_agents = data.get("active_agents")
        if isinstance(active_agents, (list, set, tuple)):
            base_key = _extract_base_agent_key(agent_id)
            if base_key not in active_agents:
                # No-op: return state unchanged (and no new message)
                return {"messages": state["messages"], "data": state["data"]}
        # Otherwise run the underlying function
        return agent_function(state, agent_id=agent_id)

    return maybe_skip