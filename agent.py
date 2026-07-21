from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from langgraph.types import Send
import json

llm = ChatOllama(model="llama3.2")

class State(TypedDict):
    query: str
    currentPlanningData: list[tuple[str, str]]
    result: str
    response: Annotated[list[str], operator.add]

available_actions = [("Retrieve", "Retrieve user information from the knowledge base. Pick this if we have not collected enough data to answer the question."), ("Generate", "Generate the final answer. Pick this once we have enough data to answer the final question.")]

def planner(state: State):
    original_query = state["query"]
    dataStr = ""
    actions_executed = []
    if "currentPlanningData" in state:
        for action, result in state["currentPlanningData"]:
            dataStr += f"Action: {action}, Result: {result}\n"
            actions_executed.append(action)

    available_actions_str = ""
    for action, description in available_actions:
        if action in actions_executed:
            continue
        available_actions_str += f"{action}: {description}\n"

    prompt = f"""
    You are a planner helping to answer the following question from customer: {original_query}

    Here is the data that has already been collected:
    {dataStr}
    Here, `Action` refers to action that was executed, and `Result` refers to the result of that action.

    Please decide which action to take next. You can choose from the following actions:
    {available_actions_str}

    Please return a JSON object with the following format:
    {{"action": "[name of action]"}}
    ONLY return this object, don't return anything else. Note that the name of the action should only be one of the following: ["Retrieve", "Generate"]. DO NOT PICK ANY ACTION ALREADY EXECUTED. These are the actions that have already been executed: {actions_executed}

    Remember, ONLY return the json object. DO NOT RETURN ANYTHING ELSE.
    """
    response = llm.invoke(prompt)
    return {
        "response": [response.content]
    }

def retrieve():
    return f"""Subscription plan: $190 per month"""

def generate_final_answer():
    print("generate final answer called!")

actionToFunction = {"Retrieve": retrieve, "Generate": generate_final_answer}
def executor(state: State):
    print(state["response"][-1])
    response = json.loads(state["response"][-1])
    action_name = response["action"]
    if action_name == "Generate":
        state["result"] = "done"
        return state
    f = actionToFunction[action_name]
    data = f()
    if "currentPlanningData" not in state:
        state["currentPlanningData"] = []
    state["currentPlanningData"].append((action_name, data))
    state["result"] = "not done"
    return state

def answerer(state: State):
    dataStr = ""
    if "currentPlanningData" in state:
        for action, result in state["currentPlanningData"]:
            dataStr += f"{result}\n"
    prompt = f"""
    You are a customer support agent answering this question from a user: {state['query']}

    Here is the data that has been collected: {dataStr}

    Please provide a final answer to the user's question based on the data collected. Be concise and clear in your response.
    """
    response = llm.invoke(prompt)
    return {
        "response": [response.content]
    }

def route_after_executor(state: State):
    if state["result"] == "done":
        return "answerer"
    else:
        return "planner"

graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("answerer", answerer)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_conditional_edges("executor", route_after_executor, {"planner": "planner", "answerer": "answerer"})
graph.add_edge("answerer", END)
app = graph.compile()
result = app.invoke({
    "query": "How much money am I currently being charged for my subscription?"
})
print(result["response"][-1])


