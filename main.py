from pydantic import BaseModel
from crewai.flow import Flow, start, listen, router, and_, or_

class State(BaseModel):
  area:str=" "
  domain:str=" "
  companies:str=" "

import nest_asyncio
nest_asyncio.apply()

class Lead_Synapse(Flow[State]):

  @start()
  def data_ingest(self):
    self.state.area=input("Enter the area: ")
    self.state.domain=input("Enter the company domain: ")

  @listen(data_ingest)
  def company_scout(self):
    response=company_scout_bot().crew().kickoff({
        "area":self.state.area,
        "domain":self.state.domain
    })
    self.state.companies=response.raw

  @listen(company_scout)
  def linked_intel(self):
    response=linked_intel().crew().kickoff({"list_of_companies":self.state.companies})
    print(response.raw)

flow=Lead_Synapse()
