#!/usr/bin/env python
# src/latest_ai_development/main.py
import sys
from project_planning.crew import ProjectPlanningCrew
from datetime import datetime

def run():
  """
  Run the crew.
  """
  project = 'Website'
  industry = 'Technology'
  project_objectives = 'Create a website for a small business'
  team_members = """
  - John Doe (Project Manager)
  - Jane Doe (Software Engineer)
  - Bob Smith (Designer)
  - Alice Johnson (QA Engineer)
  - Tom Brown (QA Engineer)
  """
  project_requirements = """
  - Create a responsive design that works well on desktop and mobile devices
  - Implement a modern, visually appealing user interface with a clean look
  - Develop a user-friendly navigation system with intuitive menu structure
  - Include an "About Us" page highlighting the company's history and values
  - Design a "Services" page showcasing the business's offerings with descriptions
  - Create a "Contact Us" page with a form and integrated map for communication
  - Implement a blog section for sharing industry news and company updates
  - Ensure fast loading times and optimize for search engines (SEO)
  - Integrate social media links and sharing capabilities
  - Include a testimonials section to showcase customer feedback and build trust
  """

  # Format the dictionary as Markdown for a better display in Jupyter Lab
  # formatted_output = f"""
  # **Project Type:** {project}

  # **Project Objectives:** {project_objectives}

  # **Industry:** {industry}

  # **Team Members:**
  # {team_members}
  # **Project Requirements:**
  # {project_requirements}
  # """

  inputs = {
    'start_date': datetime.now().strftime("%Y-%m-%d"),
      'project_type': project,
      'project_objectives': project_objectives,
      'industry': industry,
      'team_members': team_members,
      'project_requirements': project_requirements
  }
  # inputs = {
  #   'topic': 'AI Agents'
  # }
  ProjectPlanningCrew().crew().kickoff(inputs=inputs)
