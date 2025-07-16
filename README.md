# DEEP RESEARCH 0.1

## 1 | HIGH‑LEVEL WORKFLOW
User prompt – The user states what they want to research.

Plan generation – The assistant (model = openai/o3‑mini, which understands complex topics and designs coherent multi‑step strategies better than lighter models like 4o) produces a research plan containing up to plan_depth steps.

Step execution – For each step, the assistant  
• runs a Google‑backed DuckDuckGo search,  
• selects up to search_depth websites to read,  
• extracts text with Trafilatura, and  
• takes detailed notes block‑by‑block.

Plan revision – After every step, the assistant may adjust the remaining plan in light of new insights or poor results.

Report writing – When all notes are gathered, the assistant drafts section titles, assigns relevant steps to each section, writes the body, and finally composes an introduction and a conclusion.

## 2 | WHY AN EXPLICIT RESEARCH PLAN?
Compared to Q&A loops: Some open‑source “deep‑research” agents collect information and follow up questions recursively, generating a tree of information. That works for narrow inquiries (e.g. “Are cats or dogs better pets?”) but fails for breadth‑heavy tasks (“List the 50 most debated political topics”). Furthermore, the Q&A style often drifts into niche, irrelevant sub‑questions.

In the research plan approach, the assistant outputs a search query and reasoning behind the query for each step, forcing the LLM to think globally and stay relevant.

## 3 | RESEARCH PIPELINE
**Step 1 — Search**  
• For the current query the agent issues a DuckDuckGo request with the Google backend and captures only the first five results. This keeps noise low and ensures all URLs are high‑quality.

**Step 2 — Site selection**  
• The LLM reviews the five links and selects up to search_depth websites to read.

**Step 3 — HTML → clean text**  
• Each chosen URL is fetched and streamed through Trafilatura.  
– Trafilatura consistently outperformed other open‑source extractors (e.g. readability.Document.summary), which often returned empty strings.  
• The first 5 000 words of clean text are retained; anything beyond that is discarded to cap cost and context.

**Step 4 — Chunking & note‑taking**  
• The 5 000‑word slice is split into overlapping blocks (≈ 600 words with a 50‑word overlap).  
• The LLM reads one block at a time and either:  
– takes detailed notes (plain text), or  
– outputs “…” to skip.  
• Reading piece‑by‑piece yields richer, context‑preserved notes than sending the full article at once.  
– Earlier RAG experiments with 600‑word retrievals often grabbed irrelevant fragments or stripped away context.  
– RAG does excel at pinpoint facts < 300 words, but for broad‑topic note‑taking the straight block method proved superior.

**Step 5 — Plan revision loop**  
• After finishing a website (or abandoning it early), the agent decides whether to rewrite the remaining plan:  
– If new insights suggest better queries, or if the previous step produced weak information, the tail of the plan is regenerated.  
– Otherwise the agent proceeds unchanged.

**Step 6 — Iterate until done**  
• The pipeline repeats for every plan step until notes are gathered for all queries.

## 4 | REPORT‑GENERATION PIPELINE
**Challenge**  
• Using a single prompt with all notes caused the response to be short and miss key details.  
• Solution: break the report down into sections and progressively assemble a full, detail‑rich document.

**Step 1 — Section titles**  
• The LLM examines the complete research plan plus the accumulated notes and returns the sections for the report, not including the introduction and conclusion.

**Step 2 — Step‑to‑section mapping**  
• For each title the agent decides which research steps contain the most relevant material.  
• Skipping this mapping overwhelms the model’s context window, causing key details to be overlooked.

**Step 3 — Section drafting**  
• The assistant receives the filtered notes and writes the section in a professional, detailed style.

**Step 4 — Introduction and Conclusion**  
• After the body is complete the agent sees the entire draft and composes an introduction and conclusion.

**Step 5 — Assembly**  
• Titles, introduction, body sections, and conclusion are concatenated into the final report and returned to the user.

## 5 | FUTURE WORK
### End‑to‑End Tool Integration
Build a unified research tool that wraps DuckDuckGo search (Google backend) and Trafilatura extraction.

Shift the plan from concrete search queries to abstract objectives (“Find recent peer‑reviewed papers on X”).

Let the assistant invoke the tool directly for each objective, rather than splitting planning and execution.

Implementation path: adopt a ReACT‑style framework.  
• Hand‑craft a small set of example dialogues showing the agent using the tool.  
• Generate additional dialogues with an LLM.  
• Train the assistant to use the tool via few‑shot prompting (augmented with RAG for example retrieval) or reinforcement learning.

### Controlled Hyperlink Exploration
Enable the agent to follow in‑article hyperlinks when they promise valuable context.

Past attempts led to over‑exploration and irrelevant notes.

Mitigation strategy: include well‑curated hyperlink examples in the training set and apply RL‑based reward shaping to discourage unnecessary link traversals while preserving the ability to dive deeper when justified.

## How to use

1. **Clone the repo**

    git clone https://github.com/ethannigrin/Deep-Research-0.1.git
    cd Deep-Research-0.1

2. **(Optional) create & activate a virtual environment**

    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # macOS/Linux

3. **Install dependencies**

    pip install -r requirements.txt

4. **Create a .env file with your OpenAI key**

    echo "OPENAI_API_KEY=" > .env       # Windows
    echo 'OPENAI_API_KEY=' > .env       # macOS/Linux

5. **Add your prompt**

    Open `main.py` and replace the `user_prompt` placeholder with your own text.

6. **Run the program**

    python main.py