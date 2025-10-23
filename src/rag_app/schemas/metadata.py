from pydantic import BaseModel


class QuestionMetadata(BaseModel):
    questions: list[str]