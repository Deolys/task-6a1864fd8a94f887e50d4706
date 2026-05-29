import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_chat_agent
from langchain_openai import ChatOpenAI

# 1. Define state
class PlanningState(TypedDict):
    task: str
    plan: List[str] | None
    current_step: int
    results: List[str]

# 2. LLM instance (expects OPENAI_API_KEY env var)
lm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

# 3. Planning node
async def planning(state: PlanningState) -> PlanningState:
    task = state["task"]
    prompt = (
        "Разбей задачу на 3–6 конкретных шагов и верни их в виде JSON массива строк.
        Пример ответа:\n\n"
        "{\n  \"plan\": [\n    \"Шаг 1: ...\",\n    \"Шаг 2: ...\"\n  ]\n}\n"
    )
    response = await lm.ainvoke(prompt + f"\nЗадача: {task}")
    text = response.content
    # Try to extract JSON
    import json, re
    try:
        data = json.loads(text)
        plan = data.get("plan", [])
    except Exception:
        # Fallback: parse numbered list
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        plan = []
        for line in lines:
            m = re.match(r"^\d+\.\s*(.*)$", line)
            if m:
                plan.append(m.group(1))
    return {
        "task": task,
        "plan": plan,
        "current_step": 0,
        "results": [],
    }

# 4. Execution node
async def execution(state: PlanningState) -> PlanningState:
    step_idx = state["current_step"]
    if state["plan"] is None or step_idx >= len(state["plan"]):
        return state
    step_text = state["plan"][step_idx]
    # Execute the step via LLM
    prompt = f"Выполни следующий шаг: {step_text}\nОтвет в виде короткого текста." 
    response = await lm.ainvoke(prompt)
    result = response.content.strip()
    new_results = state["results"] + [result]
    return {
        "task": state["task"],
        "plan": state["plan"],
        "current_step": step_idx + 1,
        "results": new_results,
    }

# 5. Condition node
def should_continue(state: PlanningState) -> str:
    if state["current_step"] >= len(state.get("plan", [])):
        return "finish"
    return "execute"

# 6. Build graph
builder = StateGraph(PlanningState)
builder.add_node("planning", planning)
builder.add_node("execution", execution)
builder.add_conditional_edges(
    "planning",
    lambda _: "execute" if _.get("plan") else END,
)
builder.add_edge("execution", "should_continue")
builder.add_conditional_edges(
    "should_continue",
    should_continue,
    {"execute": "execution", "finish": END},
)
# Entry point
builder.set_entry_point("planning")
graph = builder.compile()

# 7. Run example
if __name__ == "__main__":
    task_description = "Сравни Python и JavaScript"
    initial_state: PlanningState = {
        "task": task_description,
        "plan": None,
        "current_step": 0,
        "results": [],
    }
    result = graph.invoke(initial_state)
    # Compile final summary
    plan_steps = result["plan"] or []
    print(f"\nЗадача: {task_description}\n")
    print("План:")
    for i, step in enumerate(plan_steps, 1):
        print(f"{i}. {step}")
    print("\n[Шаги]")
    for i, res in enumerate(result["results"], 1):
        print(f"[{i}] {res}\n")
    summary = "\nИтог: " + ". ".join(result["results"])
    print(summary)
