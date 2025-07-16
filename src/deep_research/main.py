# ▸ Goal: given a user research prompt, build a multi‑step research plan, execute
#   the plan (search → choose sites → read text blocks → take notes) and finally
#   generate a multi‑section report (intro, body, conclusion).

from typing import List, Tuple, Dict, Any, Optional

from pydantic import BaseModel, Field
from pydantic_core import ValidationError

from crewai import LLM                      # thin wrapper around OpenAI models
from ddgs import DDGS                       # DuckDuckGo Search (Google backend)

import trafilatura                         # HTML → readable text extractor
from lxml import etree

# ---------------------------------------------------------------------------
# Pydantic schemas — EVERY LLM RESPONSE IS VALIDATED AGAINST ONE OF THESE (except for plain text).
# ---------------------------------------------------------------------------

class ResearchPlan(BaseModel):
    """List of research steps: (search query, reasoning)."""
    plan: List[tuple[str,str]]

class Site(BaseModel):
    """Return value when the LLM selects which search‑result URL to open."""
    site: tuple[int,str]  # (1‑based index, reasoning)

class ExploreDecision(BaseModel):
    """Should we keep reading the current URL or abandon it?"""
    decision: tuple[int,str]  # 1 = continue, 0 = abandon

class Sections(BaseModel):
    """Top‑level headings for the final report (excluding intro/conclusion)."""
    sections: List[str]

class Step_Indices(BaseModel):
    """Which research steps are relevant to a given section title."""
    step_indices: List[int]

# ---------------------------------------------------------------------------
# Text‑processing helpers
# ---------------------------------------------------------------------------


def parse_search_results(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        title   = r.get("title",  "").replace("\\n", "\n")
        url     = r.get("href",   "")
        snippet = r.get("body",   "").replace("\\n", "\n")

        lines = [f"Result {i}: {title}", f"  URL: {url}"]
        if snippet:
            lines.append(f"  Snippet: {snippet}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)

# def parse_hrefs(hrefs: list[str], start: int = 1) -> str:
#     blocks: list[str] = []
#     for i, url in enumerate(hrefs, start):
#         lines = [f"Link {i}:", f"  URL: {url}"]
#         blocks.append("\n".join(lines))

#     return "\n\n".join(blocks)

def parse_notes(notes: dict[str, str]) -> str:
    if not notes:
        return ""

    parts = []
    for url, text in notes.items():
        text = text.strip() or "(no notes)"
        block = f"{url}\n"                      \
                f"{'-'*64}\n"                   \
                f"{text}"
        parts.append(block)

    return "\n\n".join(parts)

def parse_plan_and_notes(research_plan: List[tuple[str,str]], notes: List[Dict[str, str]]):
    result = ""
    for i in range(len(research_plan)):
        result = result + f"Step {i+1}: {research_plan[i][0]}. Reason: {research_plan[i][1]}\n\n"
        result = result + f"Notes:\n{parse_notes(notes[i])}\n\n"
    return result


def parse_section_notes(step_indices: List[int], notes: List[Dict[str, str]]) -> str:
    lines: List[str] = []

    for idx in step_indices:
        if 0 <= idx < len(notes):
            step_notes = notes[idx]
            for url, text in step_notes.items():
                lines.append(f"{url}:\n{text.strip()}")
                lines.append("")

    return "\n".join(lines).rstrip()

def merge_notes(notes_a: dict[str, str], notes_b: dict[str, str], separator: str = "\n\n") -> dict[str, str]:
    merged = notes_a.copy()

    for url, note in notes_b.items():
        if url in merged and note.strip():
            if merged[url].strip():
                merged[url] += separator + note.strip()
            else:
                merged[url] = note.strip()
        else:
            merged[url] = note.strip()

    return merged


def extract_blocks(url: str, max_words_per_block: int = 600, word_overlap: int = 50, max_words_total: int = 5000) -> Tuple[List[str], List[str]]:
    # Fetch *url*, return list of overlapping text blocks and unique hrefs.

    # Trafilatura reliably extracts main‑content text even on messy pages.
    # Cap at 5 000 words to control token costs and split into ~600‑word blocks
    # with 50‑word overlap so no sentence context is lost between blocks.

    html = trafilatura.fetch_url(url)
    if html is None:
        return [], []

    text = trafilatura.extract(html, include_links=False, output_format="txt") or ""
    xml_str = trafilatura.extract(html, include_links=True, output_format="xml")

    # Collect outgoing links once so the agent can optionally explore them.
    hrefs_seen: set[str] = set()
    hrefs_list: List[str] = []

    if xml_str:
        root = etree.fromstring(xml_str.encode())
        for ref in root.xpath(".//ref[@target]"):
            target = ref.get("target")
            if target and target not in hrefs_seen:
                hrefs_seen.add(target)
                hrefs_list.append(target)

    # Sliding‑window block generation.
    words = text.split()
    stride = max(1, max_words_per_block - word_overlap)
    blocks: List[str] = []

    for start in range(0, min(max_words_total, len(words)), stride):
        chunk = " ".join(words[start : start + max_words_per_block])
        if chunk:
            blocks.append(chunk)

    return blocks, hrefs_list

# ---------------------------------------------------------------------------
# explore_page — interactive per‑URL reading loop.
# ---------------------------------------------------------------------------

def explore_page(explore_messages: list, site_url: str, site_reasoning: str):
    # Interactively read *site_url* block‑by‑block asking the LLM for notes.

    notes = ""
    blocks, hrefs = extract_blocks(site_url)
    for block_idx, text in enumerate(blocks):
        # Ask the LLM whether to take notes on this block.
        explore_messages.append(
            {"role": "user", "content": f"We are reading from {site_url} for this reason: {site_reasoning}. This is piece of text #{block_idx} from the site:\n"
                f"{text}\n\n"
                "Would you like to take notes? "
                "If you would, output the notes as plain text, and do not include any other text"
                "If you do not want to take notes, simply output \"...\" with no other text"
            }
        )
        subnotes = call_llm(explore_messages).strip()
        explore_messages.append({"role": "assistant", "content": subnotes})
        notes = notes + "\n\n" + subnotes
        
        if subnotes == "...":
            # Offer the agent a chance to abandon the entire page early.
            explore_messages.append(
                {"role": "user", "content": "Would you like to move onto the next peice of text or abandon the site?"
                    "Respond with only valid JSON matching the schema: "
                    "{\"decision\": [int, str]}\n"
                    "• **int** – 1 to keep reading, 0 to abandon the site.\n"
                    "• **str** – Your reasoning for the decision.\n"
                    "Do not include any extra text."}
            )
            raw_decision = call_llm(explore_messages)
            explore_messages.append({"role": "assistant", "content": raw_decision})
            decision = ExploreDecision.model_validate_json(raw_decision).decision[0]
            if decision == 0:
                break
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
                {"role": "user", "content": user_prompt +
                "\nGenerate a research plan. "
                "The research plan is a sequence of searches you will make to the internet. "
                "Each step has 1 search query and a reasoning for the step. "
                "Respond with only valid JSON matching the schema: "
                "{\"plan\": [[str, str]]}. Do not include any extra text."})
            raw_plan = call_llm(messages)
            messages.append({"role": "assistant", "content": raw_plan})
            research_plan = ResearchPlan.model_validate_json(raw_plan).plan
        # 2. optional plan revision
        else:
            # Supply the LLM with context about the previous step’s results
            messages.append(
                {"role": "user", "content": user_prompt + 
                 f"\nYour previous research query was {research_plan[plan_idx-1][0]}.\n"
                 f"This was your reasoning for the query: {research_plan[plan_idx-1][1]}"
                 f"These are your notes from the previous step:\n{parse_notes(notes[plan_idx-1])}\n\n"
                f"The rest of the research plan is: {research_plan[plan_idx:]}\n"
                "Would you like to revise the plan?"
                "If you found new information that gives insight into what you should actually research next, you should revise the plan. "
                "If you received information that did not give insight into what you were hoping to learn, you should also revise the plan to try again. "
                "If not, you should keep the current plan. "
                "If you want to revise the rest of the plan, please provide a new version of the rest of the plan. "
                "The research plan is a sequence of internet searches; each step contains exactly one search query and its reasoning. "
                f"You may use up to {min(plan_depth - plan_idx, 5)} more steps.\n"
                "Respond with only valid JSON matching the schema: "
                "{\"plan\": [[str, str]]}\n"
                "• plan[i][0]: **str** – The search query for the step.\n"
                "• plan[i][1]: **str** – Your reasoning for the query.\n"
                "If you can only use 1 more step, assure that plan is [[str, str]], not [str, str].\n"
                "If you would not like to revise the rest of the plan, return [[\"No\", \"Reasoning behind No\"]] as the value for \"plan\".\n"
                "Do not include any extra text."})
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
                
            # If the plan was revised, splice it into the original plan
            if new_plan[0][0].lower() != "no":
                research_plan = research_plan[0:plan_idx] + new_plan
        
        notes.append({})   # Prepare an empty dict to store this step’s notes

        # Run a DuckDuckGo search with Google backend and capture the first 5 results
        results = DDGS().text(research_plan[plan_idx][0], backend="google", max_results=5)

        # Explore up to `search_depth` URLs chosen by the LLM
        for _ in range(search_depth):
            explore_messages = messages.copy()
            explore_messages.append(
                {"role": "user", "content": f"I have searched \"{research_plan[plan_idx][0]}. "
                "These are the results:\n"
                f"{parse_search_results(results)}\n\n"
                "Which of these websites would you like to explore? "
                "Respond with only valid JSON matching the schema: "
                "{\"site\": [int, str]}\n"
                "• **int** – the number of the website you want to explore.\n"
                "• **str** – Your reasoning for the choice.\n"
                "Do not include any extra text."
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

# ---------------------------------------------------------------------------
# call_llm — wrapper around crewai.LLM with basic retry & context trimming.
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# write_report — convert plan & notes into a structured written report.
# ---------------------------------------------------------------------------

def write_report(user_prompt: str, research_plan: List[tuple[str,str]], notes: List[Dict[str, str]]):

    # Orchestrates creation of a full report from research notes.

    # Step‑by‑step:
    #  1. Ask the LLM to propose section headings.
    #  2. For each section, request which research steps to reference.
    #  3. Draft each section using its reference notes.
    #  4. Generate introduction and conclusion.
    #  5. Assemble and return the fully‑titled report as plain text.
    
    messages = []
    
    # 1) SECTION OUTLINE
    messages.append({"role": "user", "content": user_prompt + 
                     f"These are my research steps and associated notes:\n {parse_plan_and_notes(research_plan, notes)}\n\n"
                     "Now I want to write a report. What should the sections of my report be, not including the introduction and conclusion?\n"
                    "Respond with only valid JSON matching the schema: "
                    "{\"sections\": [str]}\n"
                    "Do not include any extra text"})
    raw_sections = call_llm(messages)
    
    # Validate until JSON is correct
    while True:
        try:
            sections = Sections.model_validate_json(raw_sections).sections
            messages.append({"role": "assistant", "content": raw_sections})
            break
        except ValidationError:
            raw_sections = LLM(model="openai/o3-mini").call(messages=[{
                "role": "user",
                "content": (
                    "Output this:\n"
                    f"{raw_sections}\n"
                    "correctly in the valid JSON schema: {\"sections\": [str]}\n"
                    "Do not include any other text"
                )
            }])

    # 2) MAP SECTIONS → RESEARCH STEPS
    reference_steps_for_sections = []
    for i, section in enumerate(sections):
        section_messages = messages.copy()
        section_messages.append({"role": "user", "content": 
                                 f"Which research step(s) should I reference for section {i}, \"{section}\"?\n"
                                "Respond with only valid JSON matching the schema: "
                                "{\"step_indices\": [int]}\n"
                                "Do not include any extra text"})
        raw_step_indices = call_llm(section_messages)
        # Validate until JSON is correct
        while True:
            try:
                step_indices = Step_Indices.model_validate_json(raw_step_indices).step_indices
                break
            except ValidationError:
                raw_step_indices = LLM(model="openai/o3-mini").call(messages=[{
                    "role": "user",
                    "content": (
                        "Output this:\n"
                        f"{raw_step_indices}\n"
                        "correctly in the valid JSON schema: {\"step_indices\": [int]}\n"
                        "Do not include any other text"
                    )
                }])
        reference_steps_for_sections.append(step_indices)
    
    # 3) DRAFT EACH SECTION
    written_sections = []
    for i, section in enumerate(sections):
        messages.append({"role": "user", "content": f"Write section {i}, \"{section}\"."
                         "Reference these notes:\n"
                         f"{parse_section_notes(reference_steps_for_sections[i], notes)}\n\n"
                         "Write in a professional, detailed, and clear style. Include all details from the notes relevant to the section, even if it makes the section long. Do not include any other text. Output plain text"
                         })
        written_section = call_llm(messages).strip()
        messages.append({"role": "assistant", "content": written_section})
        written_sections.append(written_section)
    
    # 4) INTRODUCTION & CONCLUSION
    report_body = "\n\n".join(written_sections)
    intro_prompt = (
        "Now write the introduction for the report. "
        "This is the report so far:\n"
        f"{report_body}\n\n"
        "Write in a professional, detailed, and clear style. "
        "Do not include any other text. Output plain text"
    )
    messages.append({"role": "user", "content": intro_prompt})
    introduction = call_llm(messages).strip()
    messages.append({"role": "assistant", "content": introduction})

    written_sections.insert(0, introduction)

    full_without_conclusion = "\n\n".join(written_sections)
    conclusion_prompt = (
        "Now write the conclusion for the report. "
        "This is the report (with introduction):\n"
        f"{full_without_conclusion}\n\n"
        "Write in a professional, detailed, and clear style. "
        "Do not include any other text. Output plain text"
    )
    messages.append({"role": "user", "content": conclusion_prompt})
    conclusion = call_llm(messages).strip()
    messages.append({"role": "assistant", "content": conclusion})

    written_sections.append(conclusion)

    # 5) ASSEMBLE TITLED SECTIONS
    titled_sections = []

    titled_sections.append("Introduction\n\n" + written_sections[0])

    for idx, section_title in enumerate(sections):
        titled_sections.append(f"{section_title}\n\n{written_sections[idx + 1]}")

    titled_sections.append("Conclusion\n\n" + written_sections[-1])

    full_report = "\n\n".join(titled_sections)
    return full_report


initial_messages = [
            {"role": "system", "content": "You are an expert researcher.\n"
                "You will first be given a prompt, to which you will generate a multi-step research plan for. Each step should consist of a search query and a reasoning behind the choosing query in that particular step.\n"
                "After you provide me the research plan, for each step, I will use your search term to browse the internet, and return a list of websites to you.\n"
                "Once you pick a website, I will present you with text from the website piece by piece. For each piece of text, I will give you the option to take notes on the text, explore a hyperlink, move onto the next piece of text, or abandon the website.\n"
                "If you believe the text is relevant to the goal of the step or groundbreaking for the overall research goal, you should take notes. If you choose to take notes, keep in mind that you cannot see the text again, so your notes should be detailed and interpretable later on.\n"
                "If you don’t believe that anything in the text is of substance, you should move onto the next piece of text and keep reading.\n"
                "If you have reviewed multiple pieces of text and most of it is irrelevant, you should abandon the site. You can explore other websites from the results I gave you earlier as long as you haven’t exhausted your limited number of site visits.\n"
                "After completing a research step, you will have the option to revise the rest of your research plan. If the information you have gained gives you insight into what your should research further, or if you did not gain as much information from the step as you intended, you should revise the rest of the plan."
                "If you were satisfied with what you gained from the step and feel the rest of the plan still applies well to the user’s request, you should keep the plan."}
        ]


if __name__ == "__main__":
    user_prompt = "I want to learn about Alpha Evolve and how it works on a deep level."
    plan_depth = 6
    search_depth = 2
    
    
    research_plan, notes = research(user_prompt, plan_depth, search_depth, initial_messages)
    for i in range(len(research_plan)):
        print(f"Step {i+1}: {research_plan[i][0]}. Reason: {research_plan[i][1]}\n\n")
        print(f"Notes:\n{parse_notes(notes[i])}\n\n")
    
    report = write_report(user_prompt, research_plan, notes)
    print(report)
