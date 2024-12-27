# from crewai import Agent, Crew, Process, Task, LLM
# from crewai.project import CrewBase, agent, crew, task

# # If you want to run a snippet of code before or after the crew starts, 
# # you can use the @before_kickoff and @after_kickoff decorators
# # https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

# llama3_2 = LLM(model="ollama/llama3.2", base_url="http://localhost:11434")

# @CrewBase
# class ResearchCrew():
# 	"""ResearchCrew crew"""

# 	# Learn more about YAML configuration files here:
# 	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
# 	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
# 	agents_config = 'config/agents.yaml'
# 	tasks_config = 'config/tasks.yaml'

# 	# If you would like to add tools to your agents, you can learn more about it here:
# 	# https://docs.crewai.com/concepts/agents#agent-tools
# 	@agent
# 	def researcher(self) -> Agent:
# 		return Agent(
# 			config=self.agents_config['researcher'],
# 			llm=llama3_2,
# 			verbose=True
# 		)

# 	@agent
# 	def reporting_analyst(self) -> Agent:
# 		return Agent(
# 			config=self.agents_config['reporting_analyst'],
# 			llm=llama3_2,
# 			verbose=True
# 		)

# 	# To learn more about structured task outputs, 
# 	# task dependencies, and task callbacks, check out the documentation:
# 	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
# 	@task
# 	def research_task(self) -> Task:
# 		return Task(
# 			config=self.tasks_config['research_task'],
# 		)

# 	@task
# 	def reporting_task(self) -> Task:
# 		return Task(
# 			config=self.tasks_config['reporting_task'],
# 			output_file='report.md'
# 		)

# 	@crew
# 	def crew(self) -> Crew:
# 		"""Creates the ResearchCrew crew"""
# 		# To learn how to add knowledge sources to your crew, check out the documentation:
# 		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

# 		return Crew(
# 			agents=self.agents, # Automatically created by the @agent decorator
# 			tasks=self.tasks, # Automatically created by the @task decorator
# 			process=Process.sequential,
# 			verbose=True,
# 			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
# 		)



from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import GithubSearchTool
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Fetch GitHub token from .env file
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Get values from environment variables
model = os.getenv("MODEL")
api_base = os.getenv("API_BASE")

# Create the LLM instance with the loaded configuration
llama3_2 = LLM(model=model, base_url=api_base)

# Initialize the GithubSearchTool
github_tool = GithubSearchTool(
    gh_token=GITHUB_TOKEN,
    content_types=['code', 'issue'],  # Options: code, repo, pr, issue
)

@CrewBase
class ResearchCrew():
    """ResearchCrew crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Add the GithubSearchTool to the researcher agent
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            llm=llama3_2,
            tools=[github_tool],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            llm=llama3_2,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the ResearchCrew crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )