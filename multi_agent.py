import os
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY

search_tool = SerperDevTool()

# os.environ["OPENAI_MODEL"] = "gpt-4-32k"   you can use this approach of nloading the model or use the below approach for the model
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Creating a senior research agent with memory and verbose mode

adhd_specialist = Agent(
    
    role = "ADHD specialist",
    goal = "Conduct thorough research to determine the level of severity from  a ADHD discharge note",
    backstory = "You are experienced ADHD specialist with expertise in finding and synthesizing information in MIMIC-IV database to detemine the severity level of ADHD patients as 'HIGH' or 'LOW' ",
    verbose = True,
    memory = True,
    tools = [search_tool],
    allow_delegation = True
)
    
    
# Creating a writer agent with custom tools and delegation capabilities 

adhd_pyschiatrists =  Agent(
    role = "ADHD Psychiatrists",
    goal = "ability to diagnose and treating mental health disorder {topic}",
    backstory = "You are experienced ADHD Pyschiatrists with expertise in diagnosing and treating mental disorder, able to prescribe medication and provide therapy, and having the detail information of  MIMIC-IV database ADHD discharge note to detemine the severity level of ADHD patients as 'HIGH' or 'LOW' ",
    verbose = True,
    memory = True,
    tools = [search_tool],
    allow_delegation = False
)
    
  # TASK 1  

adhd_specialist_task = Task(
    description = (
        "Research the following topic and provide a comprehensive and concise answer from the information: {topic}"
        "Your final context should clearly articulate the key points, "
        "identify the medication class and severity levels of each patients based of their subject id"
    ),
    expected_output = "A 5 lines of context summary of the research findings in determing the severity level of ADHD patients, inclduing key points and insights related: {topic} in formatted markdown",
    tools = [search_tool],
    agent = adhd_specialist,
    
)
    
    
adhd_pyschiatrists_task = Task(
    description = (
        "Research the following topic and provide a comprehensive and concise answer from the information: {topic}"
        "Your final context should clearly articulate the key points, "
        "identify the medication class and severity levels of each patients based of their subject id"
    ),
    expected_output = "A 5 lines of context summary of the research findings in determing the severity level of ADHD patients, inclduing key points and insights related: {topic}",
    tools = [search_tool],
    agent = adhd_pyschiatrists,
    async_execution = False,
    output_file = 'adhd_post.md',
)

# Forming the tech - focused crew with enhanced configuration
crew = Crew(
    agents = [adhd_specialist, adhd_pyschiatrists],
    tasks = [adhd_specialist_task, adhd_pyschiatrists_task],
    process = Process.sequential # Optional: Sequential task execution is default
)

result = crew.kickoff(inputs = {'topic' : 'AI in healthcare'})
print(result)