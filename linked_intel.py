from crewai import Agent, Task, Crew, Process
from crewai.project import agent, task, crew, CrewBase
from crewai_tools import SerperDevTool
from crewai.tools import tool
from exa_py import Exa

#EXA SEARCH TOOL
import os
exa_api_key = os.getenv("EXA_API_KEY")

@tool("Exa search and get contents")
def search_and_get_contents_tool(question: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""

    exa = Exa(exa_api_key)

    response = exa.search_and_contents(
        query=question,
        type="neural",
        num_results=30,
        highlights=True
    )

    parsedResult = '\n\n'.join([
        f"<Title id={idx}>{result.title}</Title>\n"
        f"<URL id={idx}>{result.url}</URL>\n"
        f"<Highlight id={idx}>{' | '.join(result.highlights)}</Highlight>"
        for idx, result in enumerate(response.results)
    ])

    return parsedResult

exa_tools = search_and_get_contents_tool

#SERPER DEV TOOL
serper_dev_tool=SerperDevTool()

#linked intel crew
@CrewBase
class linked_intel:
    """This crew will find the people from the respective companies"""

    @agent
    def linkedin_agent(self) -> Agent:
        return Agent(
            role="LinkedIn Prospector",
            goal="Find professional profiles from given companies",
            description="An expert in finding people on LinkedIn, able to search and extract names and profile URLs using web and semantic search tools.",
            tools=[exa_tools, serper_dev_tool],
            memory=True,
            verbose=True
        )

    @task
    def linkedin_task(self) -> Task:
        return Task(
            description=(
                "You will receive a {list of companies}. For each company in the list, "
                "search for at least 5-10 people who currently work there. Return their full names "
                "and LinkedIn profile links. Structure the results neatly grouped by company."
            ),
            expected_output=(
                "A markdown file 'people.md' listing each company, followed by 5-10 employee names and their LinkedIn URLs. "
                "Example:\n\n"
                "## Company: OpenAI\n"
                "- John Doe - https://www.linkedin.com/in/johndoe\n"
                "- Jane Smith - https://www.linkedin.com/in/janesmith\n"
                "...\n\n"
                "## Company: Google\n"
                "- ..."
            ),
            agent=self.linkedin_agent(),
            output_file="people.md"
        )

    @crew
    def crew(self)->Crew:
        return Crew(
            agents=[self.linkedin_agent()],
            tasks=[self.linkedin_task()]
        )
