# AI Customer Support Agent

An **AI-powered customer support agent** built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://www.langchain.com/), and [Ollama](https://ollama.com/).

The agent uses a **planner–executor–answerer architecture** to understand a customer's question, determine what information it needs, retrieve that information through available actions, and generate a final response.

The goal is to provide a foundation for an AI customer-support system that can answer questions about **customer information, subscriptions, billing, and other account-related topics** while keeping information retrieval and response generation as separate steps.

## How It Works

The agent follows an iterative workflow:

```text
                 ┌──────────────┐
                 │    START     │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │    Planner   │
                 │              │
                 │ Understands  │
                 │ the request  │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Executor   │
                 │              │
                 │ Runs the     │
                 │ selected     │
                 │ action       │
                 └──────┬───────┘
                        │
                 ┌──────┴───────┐
                 │              │
          More information   Enough information
              needed              │
                 │                ▼
                 │         ┌──────────────┐
                 └────────►│   Answerer   │
                           │              │
                           │ Generates    │
                           │ final reply  │
                           └──────┬───────┘
                                  │
                                  ▼
                               END
```

## Architecture

The agent consists of three primary components.

### 1. Planner

The planner acts as the agent's **decision-making layer**.

It receives the customer's question, conversation history, and information that has already been retrieved.

It then decides which action should be performed next.

For example, if a customer asks:

> "How much am I paying for my subscription?"

the planner can determine that it needs subscription information and select:

```json
{
  "action": "get_subscription_plan"
}
```

If the required information has already been collected, the planner can select:

```json
{
  "action": "Generate"
}
```

This tells the system that it has enough information to produce the final customer response.

### 2. Executor

The executor is responsible for **actually performing the action selected by the planner**.

Actions are mapped to Python functions:

```python
actionToFunction = {
    "Retrieve": retrieve,
    "get_subscription_plan": get_subscription_plan
}
```

This creates a controlled boundary between the LLM and the application's underlying data sources.

For example:

```python
get_subscription_plan()
```

could eventually be replaced by a call to a real billing or subscription service.

The executor stores the returned information in the agent's state so that the planner can use it during the next iteration.

### 3. Answerer

Once the agent has collected enough information, the answerer generates the final response.

It receives:

* The customer's original question
* Conversation history
* Information retrieved by the executor

The LLM then uses this information to produce a concise customer-facing answer.

This separation allows the planner to focus on **what information is needed**, while the answerer focuses on **how to communicate the answer**.

## Available Customer Support Actions

The current prototype provides three actions.

### Retrieve Customer Information

```text
Retrieve
```

Retrieves customer information from the example knowledge base, such as:

* Name
* Age

In a production system, this could connect to a customer database or CRM.

### Get Subscription Plan

```text
get_subscription_plan
```

Retrieves subscription information such as:

* Plan name
* Price
* Billing cadence
* Currency

For example:

```json
{
  "plan_name": "Premium",
  "plan_price": 9.99,
  "cadence": "monthly",
  "plan_currency": "USD"
}
```

In a production customer-support system, this could query the application's billing or subscription service.

### Generate

```text
Generate
```

Signals that the agent has gathered enough information and should generate the final customer response.

Unlike the other actions, `Generate` does not call an external function.

## Conversation Memory

The agent maintains conversation history through the `memory` state.

Each interaction is stored as:

```python
memory.append((prompt, resp))
```

The history is then provided to subsequent planner and answerer calls.

This allows the agent to use information from previous interactions rather than treating every customer question as completely independent.

For example:

```text
Customer:
What is the user data that you are storing for me?

Agent:
You have a Premium subscription...

Customer:
How much am I being charged?

Agent:
You're currently being charged $9.99 per month.
```

The second request can be processed in the context of the previous conversation.

## LangGraph Workflow

The workflow is implemented using LangGraph's `StateGraph`.

```python
graph = StateGraph(State)

graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("answerer", answerer)
```

The graph starts with the planner:

```python
graph.add_edge(START, "planner")
```

The planner always passes its decision to the executor:

```python
graph.add_edge("planner", "executor")
```

The executor then determines whether the agent needs more information:

```python
graph.add_conditional_edges(
    "executor",
    route_after_executor,
    {
        "planner": "planner",
        "answerer": "answerer"
    }
)
```

If more information is needed, the workflow returns to the planner.

If enough information has been collected, the workflow moves to the answerer.

Finally:

```python
graph.add_edge("answerer", END)
```

## State Management

The agent's state is represented using a Python `TypedDict`:

```python
class State(TypedDict):
    query: str
    currentPlanningData: list[tuple[str, str]]
    memory: list[tuple[str, str]]
    result: str
    response: Annotated[list[str], operator.add]
```

The state allows the different parts of the agent to share information during a request.

### `query`

The customer's current question.

### `currentPlanningData`

Information retrieved by actions during the current request.

### `memory`

Previous customer questions and agent responses.

### `result`

Tracks whether the agent has collected enough information.

### `response`

Stores LLM-generated planner and answerer responses.

## LLM

The agent currently uses:

```python
llm = ChatOllama(model="llama3.2")
```

This allows the customer-support agent to run using a locally hosted LLM through Ollama.

Using a local model can be useful for development and experimentation because the application does not need to depend on a hosted LLM API for this prototype.

The LLM is used for two distinct tasks:

1. **Planning** — deciding which customer-support action to execute.
2. **Response generation** — producing the final customer-facing answer.

## Example Customer Questions

The architecture can support questions such as:

```text
What information do you have about my account?
```

```text
What subscription am I currently on?
```

```text
How much am I being charged each month?
```

```text
What currency is my subscription billed in?
```

As additional customer-support actions are added, the agent can support more complex requests.

## Extending the Agent

The action-based design makes it straightforward to add additional customer-support capabilities.

Potential actions include:

```text
Get Account Details
Get Subscription
Get Billing History
Check Payment Status
Get Invoice
Update Customer Information
Cancel Subscription
Change Subscription
Check Refund Status
Create Support Ticket
Search Help Center
```

For example:

```python
available_actions = [
    ("Retrieve", "Retrieve customer account information."),
    ("get_subscription_plan", "Retrieve subscription details."),
    ("get_billing_history", "Retrieve recent billing transactions."),
    ("get_invoice", "Retrieve a customer's invoice."),
    ("Generate", "Generate the final customer response.")
]
```

Each action can then be connected to the appropriate backend service.

## Production Architecture

The current implementation is a prototype. A production version could connect the agent to real customer-support infrastructure:

```text
                    Customer
                       │
                       ▼
                ┌─────────────┐
                │ AI Support  │
                │    Agent    │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   Planner   │
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │   Actions   │
                └──────┬──────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Customer     Billing      Support
       Database     Service       System
          │            │            │
          └────────────┼────────────┘
                       ▼
                 ┌─────────────┐
                 │   Answerer  │
                 └──────┬──────┘
                        │
                        ▼
                     Customer
```

This allows the AI agent to act as an intelligent interface over existing customer-support systems.

## Security and Privacy

Because a customer-support agent may handle personal and account information, production implementations should include appropriate security controls.

Important considerations include:

* Authenticate customers before exposing account information.
* Authorize every account-specific action.
* Do not rely on the LLM to enforce permissions.
* Validate all tool inputs and outputs.
* Keep sensitive credentials outside source code.
* Avoid logging unnecessary personal information.
* Use synthetic data in examples and public repositories.
* Add safeguards around actions that modify accounts or subscriptions.

The LLM should **never be the security boundary**. Backend services should independently verify that the customer is authorized to access or modify the requested information.

## Current Limitations

This project is intended as a foundation for an AI customer-support agent rather than a production implementation.

Some areas that could be improved include:

* Structured LLM outputs instead of manually parsing JSON
* Robust error handling
* Authentication and authorization
* Persistent conversation memory
* Real database/API integrations
* Tool execution timeouts
* Retry handling
* Observability and tracing
* Guardrails against incorrect actions
* Human-agent escalation
* Confirmation before destructive actions
* Automated testing and evaluation

## Getting Started

Install the required packages:

```bash
pip install langgraph langchain-ollama
```

Install and start Ollama, then download the model:

```bash
ollama pull llama3.2
```

Run the application:

```bash
python main.py
```

The application will process the example customer questions and use the planner–executor–answerer workflow to generate responses.

