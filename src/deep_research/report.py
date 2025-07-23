from typing import List, Dict
from pydantic_core import ValidationError
from crewai import LLM 

from text_processors import (
    parse_plan_and_notes,
    parse_section_notes
)

from prompts import (
    section_drafting_prompt,
    reference_steps_for_sections_prompt,
    section_writing_prompt,
    intro_writing_prompt,
    conclusion_writing_prompt
)

from llm import (
    call_llm
)

from json_schemas import (
    Sections,
    Step_Indices
)

# ---------------------------------------------------------------------------
# write_report — convert plan & notes into a structured written report.
# ---------------------------------------------------------------------------

def write_report(user_prompt: str, research_plan: List[tuple[str,str]], notes: List[Dict[str, str]], steps_allowed_per_section = 3):

    # Orchestrates creation of a full report from research notes.

    # Step‑by‑step:
    #  1. Ask the LLM to propose section headings.
    #  2. For each section, request which research steps to reference.
    #  3. Draft each section using its reference notes.
    #  4. Generate introduction and conclusion.
    #  5. Assemble and return the fully‑titled report as plain text.
    
    messages = []
    
    # 1) SECTION OUTLINE
    messages.append({"role": "user", "content": section_drafting_prompt.format(
        user_prompt = user_prompt,
        research_steps_and_notes = parse_plan_and_notes(research_plan, notes)
    )
    }
    )
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
                    "correctly in the valid JSON schema: {\"sections\": [[str, str]]}\n"
                    "Do not include any other text"
                )
            }])

    # 2) MAP SECTIONS → RESEARCH STEPS
    reference_steps_for_sections = []
    for i, section in enumerate(sections):
        messages = []
        messages.append({"role": "user", "content": reference_steps_for_sections_prompt.format(
            user_prompt = user_prompt,
            research_steps_and_notes = parse_plan_and_notes(research_plan, notes),
            section_titles = ", ".join(title[0] for title in sections),
            section_title = section[0],
            section_description = section[1],
            steps_allowed = min(steps_allowed_per_section, len(research_plan))
        )
        }
        )
        raw_step_indices = call_llm(messages)
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
        messages = []
        messages.append({"role": "user", "content": section_writing_prompt.format(
            user_prompt = user_prompt,
            section_titles = ", ".join(title[0] for title in sections),
            current_report = "\n\n".join(written_sections),
            section_title = section[0],
            section_description = section[1],
            reference_steps_and_notes = parse_section_notes(reference_steps_for_sections[i], notes)
        )
        }
        )
        written_section = call_llm(messages).strip()
        written_sections.append(written_section)
    
    # 4) INTRODUCTION & CONCLUSION
    messages = []
    messages.append({"role": "user", "content": intro_writing_prompt.format(
        user_prompt = user_prompt,
        current_report = "\n\n".join(written_sections)
    )
    }
    )
    introduction = call_llm(messages).strip()
    written_sections.insert(0, introduction)

    messages = []
    messages.append({"role": "user", "content": conclusion_writing_prompt.format(
        user_prompt = user_prompt,
        current_report = "\n\n".join(written_sections)
    )
    }
    )
    conclusion = call_llm(messages).strip()
    written_sections.append(conclusion)

    # ----------------------------------------------------------
    # 5) APPEND SOURCE URLS AFTER EACH BODY SECTION
    # ----------------------------------------------------------
    sections_with_sources: list[str] = []

    for i, section_text in enumerate(written_sections):
        sections_with_sources.append(section_text)

        # body sections start at index 1 (0 = Intro) and end before the last
        body_idx = i - 1
        if 0 <= body_idx < len(reference_steps_for_sections):
            # gather unique URLs referenced by this section
            urls: set[str] = set()
            for step_idx in reference_steps_for_sections[body_idx]:
                if 0 <= step_idx < len(notes):
                    urls.update(notes[step_idx].keys())  # notes[idx] → {url: text} :contentReference[oaicite:0]{index=0}

            if urls:
                sources_block = "Sources:\n" + "\n".join(f"- {u}" for u in sorted(urls))
                sections_with_sources.append(sources_block)

    full_report = "\n\n".join(sections_with_sources)
    return full_report
