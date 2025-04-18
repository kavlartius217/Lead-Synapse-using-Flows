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

#The company_scout crew
@CrewBase
class company_scout_bot:
  """ This crew will return a list of companies in a specific area and a domain"""

  @agent
  def company_finder(self)->Agent:
    return Agent(
        role="Company Discovery Specialist",
        goal="Identify and extract a relevant list of companies based on a specific industry domain and geographic area for business development outreach.",
        backstory=(
            "You are a highly skilled research agent trained in identifying companies using real-time and semantic search tools. "
            "Your job is to find, evaluate, and compile a list of potential companies operating in a given sector within a specified region. "
            "Your output should be relevant, well-structured, and useful for the business development team to begin outreach."
        ),
        memory=True,
        verbose=True,
        tools=[exa_tools, serper_dev_tool]
    )

# Task definition
  @task
  def company_finder_task(self)->Task:
    return Task(
        description=(
            "Use online tools to find and extract a comprehensive list of companies that operate in the **{domain}** domain "
            "within the **{area}** region. You should use semantic and real-time search to ensure high relevance and accuracy.\n\n"
            "For each company, try to gather:\n"
            "1. Company Name\n"
            "2. Website URL\n"
            "3. Brief Description\n"
            "4. Industry tags or keywords\n"
            "5. Location (City/Country if available)\n"
            "6. Any public contact or LinkedIn URL (if accessible)\n\n"
            "The list should contain 15-20 companies that are relevant and active in the domain and location specified. "
            "Prioritize companies that are startups, scale-ups, or industry leaders."
        ),
        expected_output=(
            "A markdown file titled `companies.md` that contains a well-formatted list of 15-20 companies matching the given domain and location. "
            "Each entry should include the company name, description, website, and any additional available metadata like location or contact info. "
            "The file should be structured with headings and bullet points for easy reading by the business development team."
        ),
        agent=self.company_finder(),
        output_file="companies.md"
    )

  @crew
  def crew(self)->Crew:
   return Crew(
      agents=[self.company_finder()],
      tasks=[self.company_finder_task()]
    )

