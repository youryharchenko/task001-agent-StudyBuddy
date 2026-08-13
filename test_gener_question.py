import time

from langgraph.graph.state import RunnableConfig
from langgraph.types import Command

from agent import PlanState, app, logger_callback

STUDENT_ID = "Студент01"
TOPIC_ID = "Вектори"
TASK_TYPE = "plan"
TOPIC = "Лінійна алгебра. Вектори."

if __name__ == "__main__":
    print(f"🚀 Тест - Генерація питань - Студент: {STUDENT_ID} - Тема: {TOPIC_ID}")

    thread_id = f"{STUDENT_ID}/{TOPIC_ID}"

    start_time = time.time()
    config: RunnableConfig = {
        "recursion_limit": 25,
        "callbacks": [logger_callback],
        "configurable": {"thread_id": thread_id},
    }

    try:
        # Отримуємо поточний стан
        state = app.get_state(config)
        if state.values and state.next:
            # Випадок 1: Стан є і граф НЕ завершений (є наступний вузол)
            print(
                "🔄 Знайдено незавершений стан!\n",
                # "Відновлюємо виконання з вузла:",
                # state.next,
            )
            print(f"Всього кроків плану: {len(state.values['plan'])}")
            print(f"Виконано кроків: {state.values['completed']}")
            print(f"Перехід на вузол: {state.next}")

            # Передаємо None, щоб продовжити з місця зупинки
            final_output = app.invoke(None, config=config)

        elif state.values and not state.next:
            # Випадок 2: Стан є, але граф вже повністю завершив роботу раніше
            print("✅ Ця сесія вже була успішно завершена раніше.")
            print("Кінцевий стан:", state)
            # final_output = state.values

        else:
            # Випадок 3: Нова сесія (збереженого стану немає)
            print("🚀 Нема збереженого стану.")
            print("Спочатку виконайте 'python test_new_plan.py' .")
            # final_output = app.invoke(new_plan_input, config=config)

        elapsed_time = round(time.time() - start_time, 2)
    except Exception as e:
        print(f"\n Виконання перервано за системною помилкою або тайм-аутом: {e}")
        elapsed_time = round(time.time() - start_time, 2)
