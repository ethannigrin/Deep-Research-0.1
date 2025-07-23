## Prompts used in research, explore_page, and write_report functions.
## Prompts heavily inspired by the LangChain Open Deep Research repository,
## including style of prompts, use of tags, pushing research focus into gathering information and not writing, examples of how to draft sections


initial_messages = [
            {"role": "system", "content": "You are an expert researcher.\n"
                "You will first be given a prompt, to which you will generate a multi-step research plan for. Each step should consist of a search query and a reasoning behind the choosing query in that particular step.\n"
                "After you provide me the research plan, for each step, I will use your search term to browse the internet, and return a list of websites to you.\n"
                "Once you pick a website, I will present you with text from the website piece by piece. For each piece of text, I will give you the option to take notes on the text, explore a hyperlink, move onto the next piece of text, or abandon the website.\n"
                "If you believe the text is relevant to the goal of the step or groundbreaking for the overall research goal, you should take notes. If you choose to take notes, keep in mind that you cannot see the text again, so your notes should be detailed and interpretable later on.\n"
                "If you don’t believe that anything in the text is of substance, you should move onto the next piece of text and keep reading.\n"
                "After completing a research step, you will have the option to revise the rest of your research plan. If the information you have gained gives you insight into what your should research further, or if you did not gain as much information from the step as you intended, you should revise the rest of the plan."
                "If you were satisfied with what you gained from the step and feel the rest of the plan still applies well to the user’s request, you should keep the plan."}
        ]

initial_research_plan_prompt = """
This is the users request: {user_prompt}.

Your job is to generate a research plan for the user's request

<What is the research plan>
- The research plan is a sequence of search queries that will be made to the internet.
- Each search query should be accompanied by a reasoning for the query.
- For each research step, I will use your query to search the internet. Then you will choose websites to explore and collect information based on the user's request and your reasoning for the query.
- After you complete a research step, you will have the option to revise the rest of the research plan based on the user's request and the information you recieved.
</What is the research plan>

<Important Guidelines>
- The goal of conducting research is to get information, not to write the final report (A separate agent will be used to write the final report).
- Output the research plan with valid JSON matching the schema: {{"plan": [[str, str]]}}
- plan[i][0] should be the search query for the step.
- plan[i][1] should be your reasoning for the query.
- Do not include any extra text.
</Important Guidelines>

With all this in mind, generate the research plan.
"""

successive_research_plan_prompt = """
This was the users request: {user_prompt}.
This was your previous research query: {previous_query}.
This was your reasoning behind the previous research query: {query_reasoning}.
After using your query to conduct research, this was the information you retrieved:
{notes}

This is the rest of your research plan:
{rest_of_plan}

Your job is to review the rest of the research plan and make changes if neccesary based on the user's request and the information you recieved.

<What is the Research Plan>
- The research plan is a sequence of search queries that will be made to the internet.
- Each search query should be accompanied by a reasoning for the query.
</What is the Research Plan>

<Instructions>
- You should first decide if you would like to revise the plan.
- If the new information gives insight into what you should actually research next, or if the new information did not give insight into what you were hoping to learn, you should consider revising the plan.
- If one of these is the case, you should review the rest of the current research plan and revise it accordingly.
- Note that you are allowed to use {steps_left} more research steps. If you have limited steps left relative to how much you want to research, you should decide what is most important to research based on the user's request and adjust the rest of the plan accordingly. If you have more research steps remaining than you believe you need, you do not need to use all the remaining steps.
- If you believe the new information is fruitful and the rest of the plan is fitting for the user's request, you should output the current rest of the plan.
</Instructions>

<Important Guidelines>
- The goal of conducting research is to get information, not to write the final report (A separate agent will be used to write the final report). Consider the quality and the substance of the information, not the formatting.
- Output the rest of the plan with valid JSON matching the schema: {{"plan": [[str, str]]}}
- plan[i][0] should be the search query for the step.
- plan[i][1] should be your reasoning for the query.
- Do not include any extra text.
</Important Guidelines>

With all this in mind, output the rest of the research plan.
"""

website_choosing_prompt = """
I have searched {search_query} for this reason: {query_reasoning}.
These are the results:
{website_results}

Your job is to choose which website you would like to explore based on your reasoning for the query.

<Important Guidelines>
- Output the website you want to explore with valid JSON matching the schema: {{"site": [int, str]}}
- The **int** should be the number of the website you want to explore.
- The **str** should be your reasoning for choosing the website. You should choose the website that will be most reliable and fruitful with information relevant to your reasoning for the search query.
- Do not include any extra text
</Important Guidelines>

With all this in mind, output which website you would like to explore along with your reasoning for the choice.
"""

note_taking_prompt = """
We are reading from {site_url} for this reason: {step_reasoning}.
This is piece of text #{block_idx} from the site:
{text}

Your job is to take notes on the text

<Instructions>
- Take notes on information from the text that is relevant to the reasoning for exploring the website.
- After you read this piece of text, you will not see it again, so your notes should be detailed, clear, and understandable.
- Do not take notes that are not relevant to your reasoning for exploring the website or that that you have already taken notes on. 
- Output your notes as plain text, and do not include any other text.
- If you dont believe you should take notes on any of the text, simply output "..."
</Instructions>

With all this in mind, output your notes.
"""

section_drafting_prompt = """
This was the users request: {user_prompt}.
These are my research steps and associated notes:
{research_steps_and_notes}

Now, I want to write a report
Your job is to generate the section titles for the report and descriptions for each section.

You can structure the sections of the report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ overview of topic A
2/ overview of topic B
3/ comparison between A and B

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things
Or, you could choose to make each item in the list a separate section in the report.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to give a report you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3

<Instructions>
- Generate the section titles for the report, minus the introduction and conclusion.
- For each section title, you should write a description of what each section should be about.
- The sections all together should contain all the information from the research that is relevant to the user's request.
- Output the section titles and associated descriptions with valid JSON matching the schema {{"sections": [[str, str]]}}
- sections[i][0] should be the title of the section
- sections[i][1] should be the description of the section.
- Do not include any extra text
</Instructions>

With all this in mind, output the sections and associated decriptions
"""

reference_steps_for_sections_prompt = """
This was the users request: {user_prompt}.
These are my research steps and associated notes:
{research_steps_and_notes}

Now, I want to write a report
These are the sections of my report (not including the introduction and conclusion):
{section_titles}
I want to write the section "{section_title}", which is about this: {section_description}

Your job is to decide which research steps I should refer to when writing this section

<Important Guidelines>
- Choose the research steps which are most relevent to this section.
- Decide based on the information from the notes for each research step.
- You may choose up to {steps_allowed} research steps to reference, but only choose steps which are relevent to what the section is about.
- Output only valid JSON matching the schema: {{"step_indices": [int]}}
- The **int**'s in the list refer to the number of the research steps to reference for the section
</Important Guidelines>

With all this in mind, output the numbers of the research steps which you would like to reference for the section.
"""

section_writing_prompt = """
This was the users request: {user_prompt}.

Now, I want to help me write the report.
These are the sections of my report (not including the introduction and conclusion):
{section_titles}
This is my report so far:
{current_report}

Your job is to write the section {section_title}, which is about this: {section_description}

While writing, reference these research steps and associated notes:
{reference_steps_and_notes}

<Important Guidelines>
- Reference the notes from the research steps when writing the section, and only add context outside the notes when you are extremely confident in your validity.
- Include all details from the notes relevant to the section. Do not worry about the section being too long. Use as much detail and depth as needed.
- Only focus on details/topics that are relevant to this section, are not already written in the report, and that would not be better focused on in a future section (although it is okay to breifly mention them for context if neccesary).
- Write in a professional, detailed, and clear style.
- Output the section title, followed by 2 new lines, followed by the text for the section, all as plain text. Do not include any other text.
</Important Guidelines>

With all this in mind, write the section.

"""

intro_writing_prompt = """
This was the users request: {user_prompt}.
This is the report body:
{current_report}

Your job is to write an introduction for the report
Write in a professional and clear style.
Output "Introduction", followed by 2 new lines, followed by the text for the introduction, all as plain text. Do not include any other text.

With all this in mind, write the introduction.
"""

conclusion_writing_prompt = """
This was the users request: {user_prompt}.
This is the report so far:
{current_report}

Your job is to write a conclusion for the report
Write in a professional and clear style.
Output "Conclusion", followed by 2 new lines, followed by the text for the conclusion, all as plain text. Do not include any other text.

With all this in mind, write the conclusion.
"""