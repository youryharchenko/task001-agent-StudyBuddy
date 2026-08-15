import sys
import time

from langgraph.graph.state import RunnableConfig
from langgraph.types import Command

from agent import PlanState, app, logger_callback

STUDENT_ID = "Студент01"
TOPIC_ID = "Вектори"
TASK_TYPE = "help"
TOPIC = "Лінійна алгебра. Вектори."
QUERY = "Розкажіть, що таке вектор."

if __name__ == "__main__":
    print(f"🚀 Тест - Консультант - Студент: {STUDENT_ID} - Тема: {TOPIC_ID}")

    thread_id = f"{STUDENT_ID}/{TOPIC_ID}"

    query_input: PlanState = {
        "student_id": STUDENT_ID,
        "topic_id": TOPIC_ID,
        "task_type": TASK_TYPE,
        "topic": TOPIC,
        "completed": 0,
        "current_step_idx": 0,
        "input_text": "",
        "grade": 0,
        "messages": [QUERY],
        "past_steps": [],
        "goal": "",
        "plan": [],
        "results": [],
        "status": "",
    }

    start_time = time.time()
    config: RunnableConfig = {
        "recursion_limit": 25,
        "callbacks": [logger_callback],
        "configurable": {"thread_id": thread_id},
    }

    try:
        print(f"Питання: {QUERY}")
        final_output = app.invoke(query_input, config=config)
        print(f"Відповідь: {final_output['messages'][-1].content}")

        elapsed_time = round(time.time() - start_time, 2)
    except KeyboardInterrupt:
        print("💥 Програму зупинено користувачем.")
        sys.exit(0)
    except Exception as e:
        print(f"\n Виконання перервано за системною помилкою або тайм-аутом: {e}")
        elapsed_time = round(time.time() - start_time, 2)
