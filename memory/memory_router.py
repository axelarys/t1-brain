import logging
from memory.session_memory import PersistentSessionMemory
from agents.tool_agent import ToolAgent
from tools.tool_discovery import ToolDiscovery
from actions.schema_parser import parse_action_schema
from utils.audit_logger import log_audit_event  # ✅ New

logger = logging.getLogger("memory_router")


class MemoryRouter:
    def __init__(self):
        self.memory = PersistentSessionMemory()
        self.tools = ToolDiscovery().list_tools()
        self.agent = ToolAgent()
        logger.info("🚀 MemoryRouter initialized.")
        self.memory.connect_to_db()

    def enrich_and_classify(self, session_id: str, query: str, past_context: str = "") -> dict:
        try:
            return {
                "query": query,
                "intent": "clarify_intent",
                "emotion": "neutral",
                "topic": "general",
                "session_id": session_id
            }
        except Exception as e:
            logger.warning(f"⚠️ Enrichment failed: {e}")
            return {
                "query": query,
                "intent": "fallback",
                "emotion": "neutral",
                "topic": "unknown",
                "session_id": session_id
            }

    def execute_action(self, session_id: str, user_input: str, enriched: dict = None) -> dict:
        try:
            if not enriched:
                enriched = self.enrich_and_classify(session_id, user_input)

            schema = parse_action_schema(enriched)
            api_key = enriched.get("api_key", "unknown")  # 🔍 optional if passed

            if not schema or not schema.get("tool"):
                fallback = "I'm not sure how to help with that right now."
                self.memory.store_memory(session_id, user_input, fallback, memory_type="semantic",
                                         sentiment=enriched.get("emotion", "neutral"),
                                         metadata=enriched)
                logger.info(f"✅ Stored 'semantic' memory for session={session_id}")

                log_audit_event(
                    api_key=api_key,
                    endpoint="/tool/memory",
                    action="none",
                    session_id=session_id,
                    status="fallback"
                )

                return {
                    "status": "fallback",
                    "tool": "none",
                    "output": fallback
                }

            result = self.agent.run(schema)
            self.memory.store_memory(session_id, user_input, result.get("output", ""), memory_type="tool",
                                     sentiment=enriched.get("emotion", "neutral"),
                                     metadata=enriched)
            logger.info(f"✅ Stored 'tool' memory for session={session_id}")

            log_audit_event(
                api_key=api_key,
                endpoint="/tool/memory",
                action=schema.get("tool", ""),
                session_id=session_id,
                status=result.get("status", "success")
            )

            return {
                "status": "success",
                "tool": result.get("tool", ""),
                "output": result.get("output", "")
            }

        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")

            log_audit_event(
                api_key="unknown",
                endpoint="/tool/memory",
                action="unknown",
                session_id=session_id,
                status="error"
            )

            return {"status": "error", "message": str(e)}

    def route_user_query(self, session_id: str, user_input: str) -> dict:
        try:
            past_context = ""
            try:
                similar = self.memory.find_similar_queries(user_input)
                past_context = "\n".join([m["query"] for m in similar]) if similar else ""
                logger.info(f"[Router] 🔍 FAISS context length: {len(past_context)}")
            except Exception as faiss_err:
                logger.warning(f"⚠️ Could not fetch past_context: {faiss_err}")

            enriched = self.enrich_and_classify(session_id, user_input, past_context=past_context)
            return self.execute_action(session_id, user_input, enriched=enriched)

        except Exception as e:
            logger.error(f"❌ Route failure: {e}")
            return {"status": "error", "message": str(e)}

    def list_available_tools(self) -> list:
        return self.tools

    def run_tool_direct(self, tool_name: str, input_data: dict) -> dict:
        try:
            if tool_name not in self.tools:
                return {"status": "error", "message": f"Tool '{tool_name}' not found"}
            schema = {"tool": tool_name, "input": input_data}
            return self.agent.run(schema)
        except Exception as e:
            logger.error(f"❌ Direct tool run error: {e}")
            return {"status": "error", "message": str(e)}

    def enrich_only(self, session_id: str, user_input: str) -> dict:
        try:
            past_context = ""
            try:
                similar = self.memory.find_similar_queries(user_input)
                past_context = "\n".join([m["query"] for m in similar]) if similar else ""
            except Exception:
                pass
            return self.enrich_and_classify(session_id, user_input, past_context)
        except Exception as e:
            logger.warning(f"⚠️ Enrichment-only path failed: {e}")
            return {}
