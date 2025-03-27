import sys
sys.path.append("/root/projects/t1-brain/langchain_tools")

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI  # ✅ Modern import
from route_memory_tool import route_memory
from clarify_intent_tool import clarify_intent
from config.settings import OPENAI_API_KEY

# 🛠️ Register tools
tools = [route_memory, clarify_intent]

# 🤖 LLM Setup (GPT-4)
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,
    api_key=OPENAI_API_KEY
)

# 🧠 Agent Initialization
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)
