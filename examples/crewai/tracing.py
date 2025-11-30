"""
This is an example for leveraging MLflow's auto tracing capabilities for CrewAI.
Most codes are from https://github.com/crewAIInc/crewAI-examples/tree/main/trip_planner.

For more information about MLflow Tracing, see: https://mlflow.org/docs/latest/llms/tracing/index.html
Note that the following example works with crewai>=0.83.0.
"""

from textwrap import dedent

from crewai import Agent, Crew, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai_tools import SerperDevTool, WebsiteSearchTool

import mlflow

mlflow.set_experiment("CrewAI")

# Turn on auto tracing by calling mlflow.crewai.autolog()
mlflow.crewai.autolog()

content = "Users name is John. He is 30 years old and lives in San Francisco."
string_source = StringKnowledgeSource(content=content, metadata={"preference": "personal"})

search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()


class TripAgents:
    def city_selection_agent(self):
        return Agent(
            role="City Selection Expert",
            goal="Select the best city based on weather, season, and prices",
            backstory="An expert in analyzing travel data to pick ideal destinations",
            tools=[search_tool, web_rag_tool],
            verbose=True,
        )

    def local_expert(self):
        return Agent(
            role="Local Expert at this city",
            goal="Provide the BEST insights about the selected city",
            backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
            tools=[search_tool, web_rag_tool],
            verbose=True,
        )


class TripTasks:
    def identify_task(self, agent, origin, cities, interests, range):
        return Task(
            description=dedent(f"""
                Analyze and select the best city for the trip based
                on specific criteria such as weather patterns, seasonal
                events, and travel costs. This task involves comparing
                multiple cities, considering factors like current weather
                conditions, upcoming cultural or seasonal events, and
                overall travel expenses.
                Your final answer must be a detailed
                report on the chosen city, and everything you found out
                about it, including the actual flight costs, weather
                forecast and attractions.

                Traveling from: {origin}
                City Options: {cities}
                Trip Date: {range}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="Detailed report on the chosen city including flight costs, weather forecast, and attractions",
        )

    def gather_task(self, agent, origin, interests, range):
        return Task(
            description=dedent(f"""
                As a local expert on this city you must compile an
                in-depth guide for someone traveling there and wanting
                to have THE BEST trip ever!
                Gather information about key attractions, local customs,
                special events, and daily activity recommendations.
                Find the best spots to go to, the kind of place only a
                local would know.
                This guide should provide a thorough overview of what
                the city has to offer, including hidden gems, cultural
                hotspots, must-visit landmarks, weather forecasts, and
                high level costs.
                The final answer must be a comprehensive city guide,
                rich in cultural insights and practical tips,
                tailored to enhance the travel experience.

                Trip Date: {range}
                Traveling from: {origin}
                Traveler Interests: {interests}
            """),
            agent=agent,
            expected_output="Comprehensive city guide including hidden gems, cultural hotspots, and practical travel tips",
        )


class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.cities = cities
        self.origin = origin
        self.interests = interests
        self.date_range = date_range

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()

        identify_task = tasks.identify_task(
            city_selector_agent, self.origin, self.cities, self.interests, self.date_range
        )
        gather_task = tasks.gather_task(
            local_expert_agent, self.origin, self.interests, self.date_range
        )

        crew = Crew(
            agents=[city_selector_agent, local_expert_agent],
            tasks=[identify_task, gather_task],
            verbose=True,
            memory=True,
            knowledge={"sources": [string_source], "metadata": {"preference": "personal"}},
        )

        result = crew.kickoff()
        return result


trip_crew = TripCrew("California", "Tokyo", "Dec 12 - Dec 20", "sports")
result = trip_crew.run()
print("\n\n########################")
print("## Here is you Trip Plan")
print("########################\n")
print(result)
