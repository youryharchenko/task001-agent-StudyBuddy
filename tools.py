from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field


class GenerateQuestionInput(BaseModel):
    """Схема валідації вхідних даних для генерації тестових питань."""

    model_config = ConfigDict(extra="forbid")

    topic: str = Field(..., description="Тема тестового питання")
    difficulty: Literal["низька", "середня", "висока"] = Field(
        "середня", description="Рівень складності тестового питання"
    )


@tool("generate_question", args_schema=GenerateQuestionInput)
def generate_question(
    topic: str, difficulty: Literal["низька", "середня", "висока"]
) -> str:
    """
    Генерує тестове питання на задану тему вказаної складності.

    Args:
        topic str "Тема тестового питання"
        difficulty ("низька", "середня", "висока") "Рівень складності тестового питання"
        )

    Returns:
         str "Тестове питання"
    """
    return "Це має бути тестове питання."


class CheckAnswerInput(BaseModel):
    """
    Схема валідації вхідних даних для перевірки відповіді.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., description="Тестове питання")
    student_answer: str = Field(..., description="Відповідь студента на питання")
    correct_answer: str = Field(..., description="Правильна відповідь на питання")


@tool("check_answer", args_schema=CheckAnswerInput)
def check_answer(question: str, student_answer: str, correct_answer: str) -> str:
    """
    Перевіряє відповідь студента.

    Args:
        question str = "Тестове питання"
        student_answer str "Відповідь студента на питання"
        correct_answer str "Правильна відповідь на питання"

    Returns:
         str "Оцінка відповіді студента"
    """

    return "Це має бути оцінка відповіді студента."


class ExplainConceptInput(BaseModel):
    """
    Схема валідації вхідних даних для пояснення концепції.
    """

    model_config = ConfigDict(extra="forbid")

    concept: str = Field(..., description="Концепція")
    level: Literal["коротко", "детально"] = Field(
        ..., description="Рівень пояснення концепції"
    )


@tool("explain_concept", args_schema=ExplainConceptInput)
def explain_concept(concept: str, level: Literal["коротко", "детально"]) -> str:
    """
    Дає коротке або детальне пояснення концепції.

    Args:
         concept str "Концепція"
         level str ("коротко", "детально") "Рівень пояснення концепції"

    Returns:
         str "Пояснення концепції."
    """

    return "Це має бути пояснення концепції."


class SubmitGradeInput(BaseModel):
    """
    Схема валідації вхідних даних для виставлення оцінки в LMS.
    """

    model_config = ConfigDict(extra="forbid")

    student_id: str = Field(..., description="Ідентифікатор студента")
    assignment: str = Field(..., description="Ідентифікатор практичної роботи")
    grade: Literal["незадовільно", "задовільно", "добре", "відмінно"] = Field(
        ..., description="Оцінка практичної роботи"
    )


@tool("submit_grade", args_schema=SubmitGradeInput)
def submit_grade(
    student_id: str,
    assignment: str,
    grade: Literal["незадовільно", "задовільно", "добре", "відмінно"],
) -> str:
    """
    Виставляє оцінку в LMS.

    Args:
        student_id str "Ідентифікатор студента"
        assignment str "Ідентифікатор практичної роботи"
        grade str ("незадовільно", "задовільно", "добре", "відмінно"] "Оцінка практичної роботи"

    Returns:
        str "Результат виставлення оцінки."
    """

    return "Результат виставлення оцінки."
