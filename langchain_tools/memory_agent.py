import sys
import os
import logging

sys.path.append("/root/projects/t1-brain/langchain_tools")

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from route_memory_tool import route_memory
from clarify_intent_tool import clarify_intent
from config.settings import OPENAI_API_KEY

# 🧠 Logger Setup
LOG_DIR = "/root/projects/t1-brain/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "memory_agent.log")

agent_logger = logging.getLogger("memory_agent_logger")
agent_logger.setLevel(logging.INFO)

if not agent_logger.handlers:
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    agent_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    agent_logger.addHandler(stream_handler)

# 🛠️ Register tools
tools = [route_memory, clarify_intent]
agent_logger.info("🔧 Tools registered: route_memory, clarify_intent")

# 🤖 LLM Setup (GPT-4)
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3,
    api_key=OPENAI_API_KEY
)
agent_logger.info("✅ GPT-4 LLM initialized.")

# 🧠 Agent Initialization
try:
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True
    )
    agent_logger.info("🚀 Memory agent initialized successfully.")
except Exception as e:
    agent_logger.error(f"❌ Agent initialization failed: {str(e)}")
    agent = None
