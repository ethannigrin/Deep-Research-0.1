from typing import List, Dict
from pydantic_core import ValidationError
from crewai import LLM 
from ddgs import DDGS                       # DuckDuckGo Search (Google backend)

from json_schemas import (
    ResearchPlan,
    Site
)

from text_processors import(
    extract_blocks,
    parse_notes,
    parse_search_results
)

from prompts import (
    initial_research_plan_prompt,
    successive_research_plan_prompt,
    website_choosing_prompt,
    note_taking_prompt,
)

from llm import (
    call_llm
)

# ---------------------------------------------------------------------------
# explore_page — interactive per‑URL reading loop.
# ---------------------------------------------------------------------------

def explore_page(explore_messages: list, site_url: str, step_reasoning: str):
    # Interactively read *site_url* block‑by‑block asking the LLM for notes.

    notes = ""
    blocks, hrefs = extract_blocks(site_url)
    for block_idx, text in enumerate(blocks):
        # Ask the LLM whether to take notes on this block.
        explore_messages.append(
            {"role": "user", "content": note_taking_prompt.format(
                site_url = site_url,
                step_reasoning = step_reasoning,
                block_idx = block_idx,
                text = text)
            }
        )
        subnotes = call_llm(explore_messages).strip()
        explore_messages.append({"role": "assistant", "content": subnotes})
        notes = notes + "\n\n" + subnotes
    return explore_messages, notes, hrefs

# ---------------------------------------------------------------------------
# research() — master controller for plan generation + execution (recursive).
# ---------------------------------------------------------------------------
# Signature legend:
# • plan_depth   – maximum number of research steps the user will allow
# • search_depth – maximum number of URLs explored per step
# • messages     – running chat history (user ↔︎ LLM) for context continuity
# • research_plan – list[ (query:str, reasoning:str) ] generated/updated by LLM
# • notes        – list[ {url:str → extract:str} ] collected for each step
# • plan_idx     – index of the current step being executed (0‑based)
# ---------------------------------------------------------------------------

def research(user_prompt: str, plan_depth: int, search_depth: int, messages = [], research_plan: List[tuple[str,str]] = [], notes: List[Dict[str, str]] = [], plan_idx = 0):
    # Generate (or continue) a research plan and execute step *plan_idx*.

    # If no plan exists *or* we still have un‑executed steps, keep working.
    if len(research_plan) == 0 or plan_idx < len(research_plan):
        # 1. initial plan generation
        if len(research_plan) == 0:
            # Ask the LLM to create a step‑by‑step research plan
            messages.append(
                {"role": "user", "content": initial_research_plan_prompt.format(user_prompt = user_prompt)})
            raw_plan = call_llm(messages)
            while True:
                try:
                    research_plan = ResearchPlan.model_validate_json(raw_plan).plan
                    messages.append({"role": "assistant", "content": raw_plan})
                    break
                except ValidationError:
                    raw_plan = LLM(model="openai/o3-mini").call(messages=[{
                        "role": "user",
                        "content": (
                            "Output this:\n"
                            f"{raw_plan}\n"
                            "correctly in the valid JSON schema: {\"plan\": [[str, str]]}\n"
                            "Do not include any other text"
                        )
                    }])
        # 2. optional plan revision
        else:
            # Supply the LLM with context about the previous step’s results
            messages.append(
                {"role": "user", "content": successive_research_plan_prompt.format(
                    user_prompt = user_prompt,
                    previous_query = research_plan[plan_idx - 1][0],
                    query_reasoning=research_plan[plan_idx - 1][1],
                    notes=parse_notes(notes[plan_idx - 1]),
                    rest_of_plan=research_plan[plan_idx:],
                    steps_left=min(plan_depth - plan_idx, 5)
                )
                }
            )
            raw_new_plan = call_llm(messages)
            # Keep requesting fixes until the JSON validates
            while True:
                try:
                    new_plan = ResearchPlan.model_validate_json(raw_new_plan).plan
                    messages.append({"role": "assistant", "content": raw_new_plan})
                    break
                except ValidationError:
                    raw_new_plan = LLM(model="openai/o3-mini").call(messages=[{
                        "role": "user",
                        "content": (
                            "Output this:\n"
                            f"{raw_new_plan}\n"
                            "correctly in the valid JSON schema: {\"plan\": [[str, str]]}\n"
                            "Do not include any other text"
                        )
                    }])
                
            research_plan = research_plan[0:plan_idx] + new_plan
        
        notes.append({})   # Prepare an empty dict to store this step’s notes

        # Run a DuckDuckGo search with Google backend and capture the first 5 results
        results = DDGS().text(research_plan[plan_idx][0], backend="google", max_results=5)

        # Explore up to `search_depth` URLs chosen by the LLM
        for _ in range(search_depth):
            explore_messages = messages.copy()
            explore_messages.append(
                {"role": "user", "content": website_choosing_prompt.format(
                    search_query = research_plan[plan_idx][0],
                    query_reasoning=research_plan[plan_idx][1],
                    website_results=parse_search_results(results)
                )
                }
            )
            raw_site_idx = call_llm(explore_messages)
            # Validate until JSON is correct
            while True:
                try:
                    site_idx = Site.model_validate_json(raw_site_idx).site[0] - 1
                    explore_messages.append({"role": "assistant", "content": raw_site_idx})
                    break
                except ValidationError:
                    raw_site_idx = LLM(model="openai/o3-mini").call(messages=[{
                        "role": "user",
                        "content": (
                            "Output this:\n"
                            f"{raw_new_plan}\n"
                            "correctly in the valid JSON schema: {\"site\": [int, str]}\n"
                            "Do not include any other text"
                        )
                    }])
            explore_messages.append({"role": "assistant", "content": raw_site_idx})
            # Retrieve the selected URL, remove it from further consideration,
            # and extract the page’s content into notes[plan_idx]
            site_idx = Site.model_validate_json(raw_site_idx).site[0] - 1
            site_url = results[site_idx].get("href", "")
            results.pop(site_idx)
            _, notes[plan_idx][site_url], _ = explore_page(explore_messages, site_url, research_plan[plan_idx][1])
        # Recurse to the next step
        return research(user_prompt, plan_depth, search_depth, messages, research_plan, notes, plan_idx + 1)
    # All steps complete – return final plan & notes
    return research_plan, notes
