import operator
import sqlite3
from typing import Annotated, List, Literal, Optional, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from kb import knowledge_base
from logger import TrajectoryLogger
from plan import Plan


class PlanState(TypedDict):
    student_id: str
    topic_id: str
    input_text: str
    task_type: str
    status: str
    messages: Annotated[List, operator.add]
    goal: str
    plan: List[str]  # список кроків (план підготовки)
    results: List[str]  # результати виконаних кроків
    completed: int  # відсоток виконаного плану
    topic: str
    grade: int
    past_steps: Annotated[
        List[str], operator.add
    ]  # Історія виконаних дій та результатів
    current_step_idx: int  # Індекс поточного кроку


# ==========================================
# ЗАХИСНІ КОНФІГУРАЦІЇ
# ==========================================
MAX_ITERATIONS = 3  # Максимальна кількість спроб Generator <-> Evaluator
MAX_EVAL_STEPS = 5  # Захист max_steps для внутрішнього циклу Evaluator

# ==========================================
# 2. НАЛАШТУВАННЯ МОДЕЛЕЙ
# ==========================================
# MODEL_NAME = "qwen2.5-coder:7b"
# MODEL_NAME = "qwen3.5:9b"
MODEL_NAME = "gemma4:e4b"
# MODEL_NAME = "qwen2.5-coder:14b"
OLLAMA_SERVER_IP = "192.168.2.102"

TIMEOUT_SEC = 60  # Загальний тайм-аут у секундах

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0.1,
    num_predict=8192,
    # reasoning=False,
    base_url=f"http://{OLLAMA_SERVER_IP}:11434",
    client_kwargs={"timeout": TIMEOUT_SEC},
)


available_tools = []
llm_with_tools = llm.bind_tools(available_tools, tool_choice="any")
llm_planner = llm.with_structured_output(Plan)
risk_tools = []


tools_by_name = {t.name: t for t in available_tools}


def dispatch_node(
    state: PlanState,
) -> Command[Literal["planner", "generator", "evaluator", "helper"]]:
    """Розподіляє запити між вузлами."""

    task_type = state.get("task_type", "")
    match task_type:
        case "plan":
            return Command(goto="planner")
        case "gener":
            return Command(goto="generator", update={"task_type": "gener"})
        case "eval":
            return Command(goto="evaluator", update={"task_type": "eval"})
        case _:
            return Command(goto="helper", update={"task_type": "help"})


def plan_node(
    state: PlanState,
) -> Command[Literal["generator", "evaluator", "helper"]]:
    """Розробляє план підготовки студента."""

    topic_id = state.get("topic_id", "невідома тема")
    topic = state.get("topic", "невідома тема")

    results = knowledge_base.query(query_texts=[topic], n_results=3)
    context = ""
    if results["documents"]:
        docs = results["documents"][0]
        context = f"КОНТЕКСТ:\n{'\n---\n'.join(docs)}"

    prompt = [
        SystemMessage(
            content=(
                "Ти — методист-планувальник занять з математики.\n"
                "Склади послідовний план з 5 кроків (ОБОВ'ЯЗКОВО 5 КРОКІВ) "
                f"для підготовки студента до іспиту з математики теми '{topic}'.\n"
                "План має забезпечити послідовне переходу від базових визначень до складних застосувань\n\n"
                "Наприклад. Тема 'Математичний аналіз. Границі'\n"
                "Крок 1. Точка прямування. Односторонні границі. Невизначеності\n"
                "Крок 2. Невизначеності\n"
                "Крок 3. Методи обчислення границь\n"
                "Крок 4. Чудові границі\n"
                "Крок 5. Правило Лопіталя\n\n"
                f"{context}"
            )
        ),
        HumanMessage(content=f"Тема: '{topic_id}'"),
    ]

    plan_obj: Plan = cast(Plan, llm_planner.invoke(prompt))

    # print(f"planner_node - plan: {plan_obj}")

    # print(f"planner_node - return: {result}")

    return Command(
        goto="generator",
        update={
            "task_type": "gener",
            "goal": plan_obj.goal,
            "plan": plan_obj.steps,
            "current_step_idx": 0,
            "past_steps": [f"Складено план з {len(plan_obj.steps)} кроків."],
        },
    )


def gener_node(
    state: PlanState,
) -> Command[Literal["evaluator", "helper"]]:
    """Генерує питання на задану тему."""
    return Command(goto="evaluator", update={"task_type": "eval"})


def eval_node(
    state: PlanState,
) -> Command[Literal["generator", "helper"]]:
    """Перевіряє відповідь студента."""
    return Command(goto="generator", update={"task_type": "gener"})


def help_node(
    state: PlanState,
) -> Command[Literal["generator", "evaluator"]]:
    """Робить пояснення відповідей."""
    return Command(goto="generator", update={"task_type": "gener"})


conn = sqlite3.connect("agent_state.db", check_same_thread=False)
saver = SqliteSaver(conn)

plan_workflow = StateGraph(PlanState)

plan_workflow.add_node("dispatcher", dispatch_node)
plan_workflow.add_node("planner", plan_node)
plan_workflow.add_node("generator", gener_node)
plan_workflow.add_node("evaluator", eval_node)
plan_workflow.add_node("helper", help_node)


plan_workflow.add_edge(START, "dispatcher")

logger_callback = TrajectoryLogger()

app = plan_workflow.compile(
    checkpointer=saver, interrupt_after=["planner", "generator"]
)
