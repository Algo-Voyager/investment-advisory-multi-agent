"""Compatibility shim — import this BEFORE anything that imports `ragas`.

`ragas` 0.4.x eagerly imports `langchain_community.chat_models.vertexai`, a
module current `langchain-community` no longer ships (moved to the separate
`langchain-google-vertexai` package). This project never uses Vertex AI chat
models — Gemini goes through `langchain-google-genai` exclusively — so we
register a harmless stub in `sys.modules` instead of pulling in an unrelated
dependency or downgrading `langchain-community` (which risks breaking the
`langchain>=1.3` APIs the rest of this codebase depends on, e.g. `create_agent`).
"""

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _stub = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover — never instantiated, import-time stub only
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                "ChatVertexAI stub — this project is Gemini-only via langchain-google-genai; "
                "Vertex AI is never actually used."
            )

    _stub.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _stub
