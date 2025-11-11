from langchain_community.agent_toolkits.load_tools import load_tools
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

from dotenv import load_dotenv

load_dotenv()

llm = ChatOllama(
    model='deepseek-r1:1.5b'
)

# Example: Search + Math tools
tools = load_tools(["google-search", "llm-math"], llm=llm)

for tool in tools:
    print(f"Loaded: {tool.name}")


agent = create_react_agent(llm, tools)

agent.invoke({"user_query": "What is the square root of the latest inflation rate in the US?"})