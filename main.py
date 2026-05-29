import json
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Define the state structure
class PlanningState(TypedDict):
    task: str
    plan: List[str] | None
    current_step: int
    results: List[str]

# LLM model (ensure OPENAI_API_KEY is set in env)
llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")

# Prompt for planning node – ask for numbered list or JSON array
planning_prompt = PromptTemplate(
    input_variables=["task"],
    template=(
        "You are a helpful assistant that plans how to solve the following task.\n"
        "Task: {task}\n"
        "Return only a numbered list of 3-6 concrete steps, or a JSON array named 'plan'. Do not add any extra text."
    ),
)

# Planning node function
async def planning(state: PlanningState) -> PlanningState:
    task = state["task"]
    plan_text = await llm.invoke(planning_prompt.format(task=task))
    # Try to parse JSON first
    try:
        data = json.loads(plan_text.content)
        if isinstance(data, dict) and "plan" in data:
            steps = list(map(str.strip, data["plan"]))
        else:
            raise ValueError("JSON does not contain 'plan'")
    except Exception:
        # Fallback: parse numbered list
        lines = plan_text.content.splitlines()
        steps = []
        for line in lines:
            if line.strip():
                parts = line.split(":", 1)
                if len(parts) == 2:
                    step = parts[1].strip()
                else:
                    # try to strip leading number
                    step = line.lstrip("0123456789.\s-").strip()
                if step:
                    steps.append(step)
    return {
        "task": task,
        "plan": steps[:6],
        "current_step": 0,
        "results": [],
    }

# Execution node – run one step and record result
execution_prompt = PromptTemplate(
    input_variables=["step", "context"],
    template=(
        "You are an assistant that executes a single step of a plan.\n"
        "Step: {step}\n"
        "Context (previous results): {context}\n"
        "Return only the result of this step, no extra text."
    ),
)

async def execution(state: PlanningState) -> PlanningState:
    idx = state["current_step"]
    if idx >= len(state["plan"]):
        return state  # nothing to do
    step_text = state["plan"][idx]
    context = "\n---\n".join(state["results"]) if state["results"] else ""
    result_obj = await llm.invoke(execution_prompt.format(step=step_text, context=context))
    new_result = result_obj.content.strip()
    return {
        **state,
        "current_step": idx + 1,
        "results": state["results"] + [new_result],
    }

# Condition node – decide whether to continue or finish
async def should_continue(state: PlanningState) -> str:
    if state["current_step"] >= len(state.get("plan", [])):
        return "finish"
    return "execute"

# Build the graph
graph = StateGraph(PlanningState)
graph.add_node("planning", planning)
graph.add_node("execution", execution)
graph.add_conditional_edges(
    "planning",
    lambda _: "execute",
)
graph.add_edge("execution", "should_continue")
graph.add_conditional_edges(
    "should_continue",
    should_continue,
    {"execute": "execution", "finish": "end"},
)
# Final node – aggregate results
async def final(state: PlanningState) -> str:
    return "\n---\n".join(state["results"])

graph.set_entry_point("planning")
graph.add_node("final", final)
graph.add_edge("should_continue", "final")  # when finish

# Compile the graph into a runnable chain
chain = graph.compile()

# Demo execution function
async def run_demo(task_description: str) -> None:
    print(f"Задача: {task_description}\n")
    result = await chain.invoke({"task": task_description})
    # The final node returns a string of all results
    print("Итог:\n", result)

# If run as script, execute demo with example task
if __name__ == "__main__":
    import asyncio
    example_task = "Сравни Python и JavaScript"
    asyncio.run(run_demo(example_task))
