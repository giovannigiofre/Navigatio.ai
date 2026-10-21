"""The supervisor's last step: turn the specialists' reports into one answer."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.agents.common import build_llm_input, specialist_reports
from app.graph.state import TripState

FINAL_PROMPT = """You are the lead planner of a trip-planning team. Write the reply to the \
user from the specialists' reports.
- Consolidate the reports into one coherent plan (by topic, or by day if dates are known) and \
remove repetition.
- Use only facts from the reports. Do not add flights, hotels, prices or weather they do not \
contain.
- Keep flights and hotels framed as illustrative examples, not bookable quotes.
- If a report says data was unavailable, say so plainly instead of filling the gap.
- If there are no reports: when the request is a trip request missing key details, ask one \
short clarifying question (for example destination, dates, departure city or budget); when it \
is not about travel planning, answer briefly and say what you can help with.
Format: plain text, short section titles on their own line, '-' bullets. No Markdown headings, \
bold or tables."""

EMPTY_ANSWER_FALLBACK = (
    "Sorry, I could not put together an answer this time. Please try rephrasing your request."
)

BUDGET_NOTE = """
The team ran out of steps before finishing. Say briefly which parts of the request are not \
covered yet."""


def data_sources_footer(sources: list[str]) -> str:
    unique = list(dict.fromkeys(sources))
    if not unique:
        return ""
    return "\n\nData sources:\n" + "\n".join(f"- {source}" for source in unique)


def create_finalizer_node(llm: BaseChatModel):
    async def finalize(state: TripState) -> dict:
        messages = state["messages"]
        prompt = FINAL_PROMPT
        if state.get("finish_reason") == "step_budget":
            prompt += BUDGET_NOTE
        if not specialist_reports(messages):
            prompt += "\n\nNo specialist was consulted for this request."

        response = await llm.ainvoke(build_llm_input(prompt, messages))
        text = response.text.strip() or EMPTY_ANSWER_FALLBACK
        answer = text + data_sources_footer(state.get("sources", []))
        return {"messages": [AIMessage(content=answer)]}

    return finalize
