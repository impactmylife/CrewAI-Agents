import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

search_tool = SerperDevTool()
llm = ChatOpenAI(model="gpt-3.5-turbo")


def create_research_agent():
    return Agent(
        role = "ADHD specialist",
        goal = "Conduct thorough research to determine the level of severity from  a ADHD discharge note",
        backstory = "You are experienced ADHD specialist with expertise in finding and synthesizing information in MIMIC-IV database to detemine the severity level of ADHD patients as 'HIGH' or 'LOW' ",
        verbose = True,
        allow_delegation = False,
        tools = [search_tool],
        llm = llm
    )
    
def create_research_task(agent, topic):
    return Task(
        description = f'Research the following topic and provide a comprehensive and concise answer from the information: {topic}',
        agent = agent,
        expected_output = "A 5 lines of context summary of the research findings in determing the severity level of ADHD patients, inclduing key points and insights related: {topic}",
        
    )
    
def run_research(topic):
    agent = create_research_agent()
    task = create_research_task(agent, topic)
    crew = Crew(agents = [agent], tasks =[task])
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Welxome to the Research Agents")
    topic = input("Enter the research topic: ")
    result = run_research(topic)
    print("Research Result: ")
    print(result)