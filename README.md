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

## 2 | RESEARCH PIPELINE
**Step 1 — Search**  
• For the current query the agent issues a DuckDuckGo request with the Google backend and captures only the first five results. This keeps noise low and ensures all URLs are high‑quality.

**Step 2 — Site selection**  
• The LLM reviews the five links and selects up to search_depth websites to read.

**Step 3 — HTML → clean text**  
• Each chosen URL is fetched and streamed through Trafilatura.  
– Trafilatura consistently performed well at extracting text.  
• The first 5 000 words of clean text are retained; anything beyond that is discarded to cap cost and context.

**Step 4 — Chunking & note‑taking**  
• The 5 000‑word slice is split into overlapping blocks (≈ 600 words with a 50‑word overlap).  
• The LLM reads one block at a time and either:  
– takes detailed notes (plain text), or  
– outputs “…” to skip.  
• Reading piece‑by‑piece yields richer, context‑preserved notes than sending the full article at once.  
– Earlier RAG experiments with 600‑word retrievals often grabbed irrelevant fragments or stripped away context.  

**Step 5 — Plan revision loop**  
• After finishing a website, the agent decides whether to rewrite the remaining plan:  
– If new insights suggest better queries, or if the previous step produced weak information, the tail of the plan is regenerated.  
– Otherwise the agent proceeds unchanged.

**Step 6 — Iterate until done**  
• The pipeline repeats for every plan step until notes are gathered for all queries.

## 4 | REPORT‑GENERATION PIPELINE
**Challenge**  
• Using a single prompt with all notes caused the response to be short and miss key details.  
• Solution: break the report down into sections and progressively assemble a full, detail‑rich document.

**Step 1 — Section titles**  
• The LLM examines the complete research plan plus the accumulated notes and returns the sections titles for the report alongside section descriptions, not including the introduction and conclusion.

**Step 2 — Step‑to‑section mapping**  
• For each title and associated description, the agent decides which research steps contain the most relevant material.  
• Skipping this mapping overwhelms the model’s context window, causing key details to be overlooked.

**Step 3 — Section drafting**  
• The assistant receives the filtered notes and writes the section in a professional, detailed, and clear style.

**Step 4 — Introduction and Conclusion**  
• After the body is complete the agent sees the entire draft and composes an introduction and conclusion.

**Step 5 — Assembly**  
• Titles, introduction, body sections, and conclusion are concatenated into the final report and returned to the user.

## 5 | FUTURE WORK
### Controlled Hyperlink Exploration
Enable the agent to follow in‑article hyperlinks when they promise valuable context. Past attempts led to over‑exploration and irrelevant notes.

## 6 | ACKNOWLEDGEMENTS
Deep Research 0.1 would not exist without the work of many open‑source developers and researchers who freely share their knowledge and code. In particular I thank:

• CrewAI – for its elegant agent‑orchestration framework, which makes multi‑step research pipelines easy to express.
• Trafilatura – for consistently reliable HTML‑to‑text extraction that allows the pipeline to capture clean, context‑rich content from the web.
• DuckDuckGo Search (Google backend) – for providing high‑quality search results that keep noise low and relevance high.
• OpenAI’s o3‑mini model – whose strong reasoning ability and cost efficiency underpin plan generation, site selection, and report drafting.
• LangChain’s “Open Deep Research” repository – whose prompt style, use of step‑wise tags, and detailed approaches served as a major design inspiration.

Your generosity, ideas, and code made this project possible. Thank you.

## How to use

1. **Clone the repo**

    git clone https://github.com/ethannigrin/Deep-Research-0.1.git
    cd Deep-Research-0.1

2. **(Optional) create & activate a virtual environment**

    python -m venv .venv

    Windows:
    .venv\Scripts\activate
    
    macOS/Linux:
    source .venv/bin/activate

3. **Install dependencies**

    pip install -r requirements.txt

4. **Create a .env file with your OpenAI key**

    echo "OPENAI_API_KEY=" > .env

5. **Add your prompt**

    Open `main.py` and replace the `user_prompt` placeholder with your own text.

6. **Run the program**

    python main.py

