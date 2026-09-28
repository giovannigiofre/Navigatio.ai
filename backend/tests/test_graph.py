from langchain_core.messages import AIMessage, HumanMessage

from app.agents.supervisor import FINISH
from app.graph.workflow import build_graph, initial_state, recursion_limit_for, run_config
from tests.fakes import Call, ScriptedChatModel, reply, route


async def run(llm: ScriptedChatModel, messages, max_steps: int = 6) -> dict:
    graph = build_graph(llm)
    return await graph.ainvoke(initial_state(messages, max_steps), config=run_config(max_steps))


def specialist_names(state: dict) -> list[str]:
    return [m.name for m in state["messages"] if isinstance(m, AIMessage) and m.name]


async def test_finish_ends_the_run_without_calling_specialists():
    llm = ScriptedChatModel(script=[route(FINISH)])

    state = await run(llm, [HumanMessage("Plan me a trip")])

    assert state["finish_reason"] == "supervisor"
    assert state["step_count"] == 0
    assert specialist_names(state) == []
    assert len(llm.calls) == 1


async def test_routes_to_the_chosen_specialist_then_finishes():
    llm = ScriptedChatModel(
        script=[route("ActivityCurator"), reply("Visit Alfama."), route(FINISH)]
    )

    state = await run(llm, [HumanMessage("Things to do in Lisbon")])

    assert specialist_names(state) == ["ActivityCurator"]
    assert state["step_count"] == 1
    assert state["finish_reason"] == "supervisor"
    assert llm.calls[1].system_prompt.startswith("You are an Activity Curator")


async def test_step_budget_forces_termination():
    def always_hotels(call: Call) -> AIMessage:
        return route("HotelScout") if call.is_route else reply("Another hotel.")

    llm = ScriptedChatModel(script=always_hotels)

    state = await run(llm, [HumanMessage("Hotels in Rome")], max_steps=3)

    assert state["finish_reason"] == "step_budget"
    assert state["step_count"] == 3
    assert specialist_names(state) == ["HotelScout"] * 3
    # The 4th supervisor turn is short-circuited by the budget and never calls the model.
    assert len(llm.route_calls()) == 3


async def test_invalid_route_output_ends_the_run():
    llm = ScriptedChatModel(script=[reply("I think the flight expert should go next.")])

    state = await run(llm, [HumanMessage("Flights to Paris")])

    assert state["finish_reason"] == "router_error"
    assert specialist_names(state) == []


async def test_history_is_passed_to_the_supervisor():
    history = [
        HumanMessage("I want a weekend in Lisbon"),
        AIMessage("Sure. Which dates?"),
        HumanMessage("October 3 to 5"),
    ]
    llm = ScriptedChatModel(script=[route(FINISH)])

    await run(llm, history)

    seen = [m.content for m in llm.calls[0].messages[1:]]
    assert seen == [m.content for m in history]


def test_recursion_limit_covers_the_step_budget():
    # Supervisor + max_steps * (specialist + Supervisor) super-steps must fit.
    for max_steps in (1, 6, 20):
        assert recursion_limit_for(max_steps) >= 1 + 2 * max_steps
    assert run_config(6)["recursion_limit"] == recursion_limit_for(6)
