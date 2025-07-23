# ▸ Goal: given a user research prompt, build a multi‑step research plan, execute
#   the plan (search → choose sites → read text blocks → take notes) and finally
#   generate a multi‑section report (intro, body, conclusion).

from research import (
    research
)

from report import (
    write_report
)

from prompts import (
    initial_messages
)

if __name__ == "__main__":
    user_prompt = "I want to understand how PPO reinforcement learning works. I want to understand the formulation of policy gradients and how step by step we get to PPO"

    plan_depth = 8
    search_depth = 3
    
    
    research_plan, notes = research(user_prompt, plan_depth, search_depth, initial_messages)
    
    report = write_report(user_prompt, research_plan, notes)
    print(report)
