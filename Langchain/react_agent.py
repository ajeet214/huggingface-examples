from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='deepseek-r1:latest'
)

# Define the tools
tools = load_tools(["llm-math"], llm=llm)

# Define the agent
agent = create_react_agent(llm, tools)

# Invoke the agent
response = agent.invoke({"messages": [("human", "what is the square root of 101?")]})
print(response)
print(response['messages'][-1].content)
