from datetime import date

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.common import FINISH
from app.graph.workflow import build_graph, initial_state, recursion_limit_for, run_config
from tests.fakes import Call, ScriptedChatModel, reply, route

TODAY = date(2026, 9, 28)


async def run(llm: ScriptedChatModel, messages, max_steps: int = 6) -> dict:
    graph = build_graph(llm, today=lambda: TODAY)
    return await graph.ainvoke(initial_state(messages, max_steps), config=run_config(max_steps))


def specialists_run(state: dict) -> list[str]:
    return [m.name for m in state["messages"] if isinstance(m, AIMessage) and m.name]


def final_answer(state: dict) -> str:
    return state["messages"][-1].content


async def test_finish_ends_the_run_with_a_clarifying_answer():
    llm = ScriptedChatModel(script=[route(FINISH), reply("Where would you like to go?")])

    state = await run(llm, [HumanMessage("Plan me a trip")])

    assert state["finish_reason"] == "supervisor"
    assert state["step_count"] == 0
    assert specialists_run(state) == []
    assert final_answer(state) == "Where would you like to go?"
    assert "No specialist was consulted" in llm.calls[1].system_prompt
    assert len(llm.calls) == 2


async def test_routes_to_the_chosen_specialist_then_finishes():
    llm = ScriptedChatModel(
        script=[route("ActivityCurator"), reply("Visit Alfama."), route(FINISH), reply("Plan.")]
    )

    state = await run(llm, [HumanMessage("Things to do in Lisbon")])

    assert specialists_run(state) == ["ActivityCurator"]
    assert state["step_count"] == 1
    assert state["finish_reason"] == "supervisor"
    assert llm.calls[1].system_prompt.startswith("You are an Activity Curator")
    assert "2026-09-28" in llm.calls[1].system_prompt


async def test_step_budget_forces_termination():
    def always_hotels(call: Call) -> AIMessage:
        return route("HotelScout") if call.is_route else reply("Another hotel.")

    llm = ScriptedChatModel(script=always_hotels)

    state = await run(llm, [HumanMessage("Hotels in Rome")], max_steps=3)

    assert state["finish_reason"] == "step_budget"
    assert state["step_count"] == 3
    assert specialists_run(state) == ["HotelScout"] * 3
    # The 4th supervisor turn is short-circuited by the budget and never calls the model.
    assert len(llm.route_calls()) == 3
    assert "ran out of steps" in llm.calls[-1].system_prompt


async def test_invalid_route_output_ends_the_run():
    llm = ScriptedChatModel(
        script=[reply("I think the flight expert should go next."), reply("Sorry.")]
    )

    state = await run(llm, [HumanMessage("Flights to Paris")])

    assert state["finish_reason"] == "router_error"
    assert specialists_run(state) == []


async def test_history_is_passed_to_the_supervisor():
    history = [
        HumanMessage("I want a weekend in Lisbon"),
        AIMessage("Sure. Which dates?"),
        HumanMessage("October 3 to 5"),
    ]
    llm = ScriptedChatModel(script=[route(FINISH), reply("ok")])

    await run(llm, history)

    seen = [m.content for m in llm.calls[0].messages[1:]]
    assert seen == [m.content for m in history]


async def test_reports_are_attributed_and_consolidated_by_the_finalizer():
    llm = ScriptedChatModel(
        script=[
            route("FlightExpert"),
            reply("TAP direct, about 2h."),
            route("HotelScout"),
            reply("Stay in Chiado."),
            route(FINISH),
            reply("Your Lisbon weekend: fly TAP, stay in Chiado."),
        ]
    )

    state = await run(llm, [HumanMessage("Weekend in Lisbon from Basel, Oct 3-5")])

    flight_report, hotel_report = state["messages"][1:3]
    assert flight_report.name == "FlightExpert"
    assert hotel_report.name == "HotelScout"

    # Later agents see earlier reports under their author's name, and the
    # message list they get still ends with the user's turn.
    supervisor_call = llm.calls[4]
    assert "### FlightExpert" in supervisor_call.system_prompt
    assert "### HotelScout" in supervisor_call.system_prompt
    assert "Already consulted for this request: FlightExpert, HotelScout" in (
        supervisor_call.system_prompt
    )
    assert isinstance(supervisor_call.messages[-1], HumanMessage)

    answer = final_answer(state)
    assert answer.startswith("Your Lisbon weekend: fly TAP, stay in Chiado.")
    assert state["messages"][-1].name is None


def test_recursion_limit_covers_the_step_budget():
    # Supervisor + max_steps * (specialist + Supervisor) + Finalize super-steps must fit.
    for max_steps in (1, 6, 20):
        assert recursion_limit_for(max_steps) >= 2 + 2 * max_steps
    assert run_config(6)["recursion_limit"] == recursion_limit_for(6)
