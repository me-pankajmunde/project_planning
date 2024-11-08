# src/latest_ai_development/crew.py
import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from crewai import LLM
from project_planning.tools.custom_tool import CurrentDateTool
from gradio_client import Client
from crewai.llm import LLM_CONTEXT_WINDOW_SIZES
from typing import List, Dict, Any, Optional, Union
import logging

# llm="ollama_chat/llama3.1"
# llm="gemini/gemini-pro"


class CustomLLM(LLM):
    def __init__(
        self,
        model: str,
        api_url: Optional[str] = None,  # Add an optional URL for custom LLM API
        timeout: Optional[Union[float, int]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[int, float]] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        callbacks: List[Any] = [],
        **kwargs,
    ):
        super().__init__(model, timeout, temperature, top_p, n, stop, max_completion_tokens, max_tokens, 
                         presence_penalty, frequency_penalty, logit_bias, response_format, seed, logprobs, 
                         top_logprobs, base_url, api_version, api_key, callbacks, **kwargs)
        self.api_url = api_url
        self.client = Client(api_url, hf_token=os.getenv('HUGGINGFACE_API_KEY')) if api_url else None  # Initialize Gradio client if URL is provided

    def call(self, messages: List[Dict[str, str]], callbacks: List[Any] = []) -> str:
        if self.client:
            # Use Gradio client to send the prompt to the custom API
            prompt = "\n".join([msg["content"] for msg in messages])
            try:
                response = self.client.predict(prompt, api_name="/chat", max_tokens=self.max_tokens)  # Adjust endpoint if needed
                return response  # Return the response content
            except Exception as e:
                logging.error(f"Custom LLM API call failed: {str(e)}")
                raise
        else:
            # Fallback to the original functionality if no API URL is provided
            return super().call(messages, callbacks=callbacks)

    def supports_function_calling(self) -> bool:
        # Support function calling if custom API is provided
        return bool(self.api_url) or super().supports_function_calling()

    def supports_stop_words(self) -> bool:
        # Support stop words if custom API is provided
        return bool(self.api_url) or super().supports_stop_words()

    def get_context_window_size(self) -> int:
        # Set a custom context window size for the API if needed
        if self.api_url:
            return int(LLM_CONTEXT_WINDOW_SIZES.get(self.model, 8192) * 0.75)
        return super().get_context_window_size()




# llm=LLM(
#         model="ollama/llama3.1:8b", 
#         base_url="http://192.168.0.16:11434"
#     )

api_url = "Nymbo/Qwen-2.5-72B-Instruct"
llm = CustomLLM(model="huggingface", api_url=api_url, max_tokens=2048)

@CrewBase
class ProjectPlanningCrew():
  """ProjectPlanning crew"""

  @agent
  def project_planning_agent(self) -> Agent:
    return Agent(
      config=self.agents_config['project_planning_agent'],
      verbose=True,
      # tools=[SerperDevTool()],
      llm=llm
    )

  @agent
  def estimation_agent(self) -> Agent:
    return Agent(
      config=self.agents_config['estimation_agent'],
      verbose=True,
      # tools=[CurrentDateTool()],
      llm=llm
    )
  
  @agent
  def resource_allocation_agent(self) -> Agent:
    return Agent(
      config=self.agents_config['resource_allocation_agent'],
      verbose=True,
      # tools=[CurrentDateTool()],
      llm=llm
    )

  @task
  def task_breakdown(self) -> Task:
    return Task(
      config=self.tasks_config['task_breakdown'],
    )

  @task
  def time_resource_estimation(self) -> Task:
    return Task(
      config=self.tasks_config['time_resource_estimation'],
    )
  
  @task
  def resource_allocation(self) -> Task:
    return Task(
      config=self.tasks_config['resource_allocation'],
      output_file='output/resource_allocation.md' # This is the file that will be contain the final report.
    )

  @crew
  def crew(self) -> Crew:
    """Creates the ProjectPlanningCrew crew"""
    return Crew(
      agents=self.agents, # Automatically created by the @agent decorator
      tasks=self.tasks, # Automatically created by the @task decorator
      process=Process.sequential,
      verbose=True,
    )
