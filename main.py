import json
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

# Define the state structure
class PlanningState(TypedDict):
    task: str
    plan: List[str] | None
    current_step: int
    results: List[str]

# LLM instance (replace with your own key if needed)
lm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

# Planning node – split the task into steps
async def planning(state: PlanningState) -> PlanningState:
    prompt = (
        "You are a helpful assistant that plans how to solve a user query.
        Given the following task, output a numbered list of 3-6 concrete steps in JSON format with key 'plan'.\n"
        f"Task: {state['task']}\n"
        "Example output:\n{"plan": ["Step 1", "Step 2"]}\n"
    )
    response = await lm.ainvoke(prompt)
    try:
        data = json.loads(response.content.strip())
        plan = data.get("plan") or []
    except Exception:
        # Fallback: split by lines
        plan = [line.strip() for line in response.content.splitlines() if line.strip().startswith(\"1.\", \"2.\", \"3.\", \"4.\", \"5.\", \"6.")]
    return {
        "task": state["task"],
        "plan": plan,
        "current_step": 0,
        "results": [],
    }

# Execution node – perform one step and record result
async def execution(state: PlanningState) -> PlanningState:
    idx = state["current_step"]
    if not state.get("plan") or idx >= len(state["plan"]):
        return state
    step_text = state["plan"][idx]
    # Ask LLM to produce result for this step
    prompt = f"You are an assistant. Execute the following step and provide a concise answer:\n{step_text}\nAnswer:"  
    response = await lm.ainvoke(prompt)
    result = response.content.strip()
    new_results = state["results"] + [result]
    return {
        "task": state["task"],
        "plan": state["plan"],
        "current_step": idx + 1,
        "results": new_results,
    }

# Condition node – decide whether to continue or finish
def should_continue(state: PlanningState) -> str:
    if state.get("current_step", 0) >= len(state.get("plan") or []):
        return "finish"
    return "execute"

# Build the graph
builder = StateGraph(PlanningState)
builder.add_node("planning", planning)
builder.add_node("execution", execution)
builder.add_conditional_edges(
    "planning",
    lambda _: "execute",
)
builder.add_edge("execution", "should_continue")
builder.add_conditional_edges(
    "should_continue",
    should_continue,
    {"execute": "execution", "finish": "finalize"},
)
# Final node – combine results
async def finalize(state: PlanningState) -> str:
    summary = "\n\nResults:\n" + "\n".join(f"[Шаг {i+1}] {r}" for i, r in enumerate(state["results"]))
    return f"Task: {state['task']}\n{summary}"

builder.add_node("finalize", finalize)
builder.set_entry_point("planning")
builder.set_finish_point("finalize")

graph = builder.compile(checkpointer=MemorySaver())

# Demo execution
if __name__ == "__main__":
    task_description = "Сравни Python и JavaScript"
    result = graph.invoke({"task": task_description, "plan": None, "current_step": 0, "results": []})
    print(result)
