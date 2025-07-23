from typing import List, Tuple, Dict
import trafilatura                         # HTML → readable text extractor
from lxml import etree


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