from typing import List, Tuple, Dict, Any
from crewai import LLM


def call_llm(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    for _ in range(3):
        try:
            raw = LLM(model="openai/o3-mini").call(messages=messages)
            return raw
        except Exception as err:
            # If we exceeded the context window, trim oldest middle messages.
            err_msg = str(err).lower()

            is_ctx = "context_length_exceeded" in err_msg

            if is_ctx and len(messages) > 6:
                del messages[1:6]
                continue

            raise