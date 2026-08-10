from pydantic import BaseModel, Field


class Plan(BaseModel):
    """План підготовки до іспиту."""

    goal: str = Field(description="Головна ціль підготовки до іспиту")
    steps: list[str] = Field(
        description="Список кроків для досягнення цілі підготовки до іспиту"
    )
