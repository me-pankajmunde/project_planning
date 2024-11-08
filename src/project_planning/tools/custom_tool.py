from typing import Type
from crewai_tools import BaseTool
from pydantic import BaseModel, Field


class CurrentDateTool(BaseTool):
    name: str = "find out current date to plan the resource accordingly"
    description: str = (
        "Find out current date to plan the resources timeline for the project"
    )

    def _run(self) -> str:
        # Implementation goes here
        import datetime

        now = datetime.datetime.now()
        current_date = now.strftime("%Y-%m-%d")

        return current_date
