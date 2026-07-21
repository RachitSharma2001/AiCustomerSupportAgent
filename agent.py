from typing import Annotated, TypedDict
import operator

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from langgraph.types import Send
import json

llm = ChatOllama(model="llama3.2")

class State(TypedDict):
    query: str
    response: Annotated[list[str], operator.add]

def planner(state: State):
    original_query = state["query"]
    prompt = f"""
    You are a planner helping to answer the following question from customer: {original_query}

    Here is the data currently collected:


    Please decide which action to take next. You can choose from the following actions:
    1. "Retrieve" -> Retrieve information from the knowledge base.
    2. "Generate" -> Generate answer

    Please return a JSON array with the following format:
    [{{"action": "[name of action]"}}, ...]
    ONLY return this array, don't return anything else. Note that the name of the action should only be one of the following: ["Retrieve", "Generate"].

    Note that array has to contain ONLY one item.
    """
    response = llm.invoke(prompt)
    return {
        "response": [response.content]
    }

def retrieve():
    print("retrieval called!")

def generate_final_answer():
    print("generate final answer called!")

actionToFunction = {"Retrieve": retrieve, "Generate": generate_final_answer}
def executor(state: State):
    response = json.loads(state["response"][0])
    print(response)
    for data in response:
        action_name = data["action"]
        f = actionToFunction[action_name]
        f()


graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_edge(START, "planner")
graph.add_edge("planner", "executor")
graph.add_edge("planner", END)
app = graph.compile()
result = app.invoke({
    "query": "How much money am I currently being charged for my subscription?"
})


