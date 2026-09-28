"""Turn the API's chat history into the message list the graph runs on."""

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.schemas import ChatTurn


def build_conversation(
    history: Sequence[ChatTurn], message: str, max_turns: int
) -> list[BaseMessage]:
    """Most recent `max_turns` history turns plus the new user message.

    The result starts with a user turn and alternates roles: leading assistant
    turns (e.g. a UI greeting) are dropped and consecutive same-role turns are
    merged (e.g. a user message whose request failed, followed by a retry).
    """
    recent = list(history[-max_turns:]) if max_turns > 0 else []
    turns = [(t.role, t.content) for t in recent] + [("user", message)]
    while turns[0][0] == "assistant":
        turns.pop(0)

    merged: list[tuple[str, str]] = []
    for role, content in turns:
        if merged and merged[-1][0] == role:
            merged[-1] = (role, f"{merged[-1][1]}\n\n{content}")
        else:
            merged.append((role, content))

    return [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in merged
    ]
