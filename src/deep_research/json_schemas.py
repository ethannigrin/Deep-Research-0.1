from pydantic import BaseModel, Field
from typing import List

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
    sections: List[tuple[str,str]]

class Step_Indices(BaseModel):
    """Which research steps are relevant to a given section title."""
    step_indices: List[int]
