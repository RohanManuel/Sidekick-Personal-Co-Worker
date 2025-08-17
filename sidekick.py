import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Any, Optional, Dict, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google.generativeai.types.generation_types import StopCandidateException

load_dotenv(override=True)

# LangChain / LangGraph imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Your tools factory functions
from sidekick_tools import playwright_tools, other_tools, calendar_tools

# ---------- Graph State ----------
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    subtasks: Optional[List[str]]

# ---------- Sidekick Implementation ----------
class Sidekick:
    def __init__(self):
        self.tools: List = []
        self.browser = None
        self.playwright = None
        self.worker_llm = None
        self.planner = None
        self.research = None
        self.code = None
        self.evaluator_llm = None
        self.graph = None
        self.memory = MemorySaver()
        self.sidekick_id = str(uuid.uuid4())

    async def setup(self):
        try:
            self.tools, self.browser, self.playwright = await playwright_tools()
        except Exception as e:
            raise RuntimeError(
                f"Playwright tools failed to initialize: {e}\n"
                "Ensure Chromium is installed (e.g., `playwright install chromium`)."
            ) from e
        try:
            self.tools += await other_tools()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize other tools: {e}") from e
        try:
            self.tools += calendar_tools()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize calendar tools: {e}") from e

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in environment.")

        # ✅ Gemini models with system message conversion
        self.worker_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, convert_system_message_to_human=True
        )
        self.planner = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, convert_system_message_to_human=True
        )
        self.research = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, convert_system_message_to_human=True
        )
        self.code = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, convert_system_message_to_human=True
        )
        self.evaluator_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", google_api_key=api_key, convert_system_message_to_human=True
        )

        await self.build_graph()

    # ---------- Worker Node ----------
    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a helpful assistant.
Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Success criteria: {state['success_criteria']}
"""
        messages = list(state["messages"])
        messages.insert(0, HumanMessage(content=system_message))

        try:
            response = self.worker_llm.invoke(messages)
        except StopCandidateException:
            response = AIMessage(content="I'm unable to respond due to safety constraints. Please try rephrasing.")

        return {"messages": [response]}

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    # ---------- Tools Node ----------
    def tools_node(self, state: State) -> Dict[str, Any]:
        last_message = state["messages"][-1]
        tool_outputs = []

        for call in getattr(last_message, "tool_calls", []):
            tool_name = call["name"]
            tool_args = call.get("args", {})

            # Find the matching tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                try:
                    result = tool.func(**tool_args)
                    if asyncio.iscoroutine(result):
                        result = asyncio.run(result)
                    tool_outputs.append(AIMessage(content=f"[{tool_name}]: {result}"))
                except Exception as e:
                    tool_outputs.append(AIMessage(content=f"[{tool_name} ERROR]: {str(e)}"))
            else:
                tool_outputs.append(AIMessage(content=f"[Tool '{tool_name}' not found]"))

        return {"messages": tool_outputs}

    # ---------- Evaluator Node ----------
    def evaluator(self, state: State) -> State:
        last_response_text = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.content), ""
        )

        eval_prompt = f"""
You are an evaluator that judges task success.

Conversation:
{self.format_conversation(state['messages'])}

Success criteria:
{state['success_criteria']}

Assistant's last response:
{last_response_text}

Please respond in this format:
Feedback: <your feedback>
Success criteria met: True/False
User input needed: True/False
"""

        eval_messages = [HumanMessage(content=eval_prompt)]
        response = self.evaluator_llm.invoke(eval_messages)
        text = response.content

        feedback = ""
        success_criteria_met = False
        user_input_needed = False

        for line in text.splitlines():
            if line.startswith("Feedback:"):
                feedback = line[len("Feedback:"):].strip()
            elif "Success criteria met:" in line:
                success_criteria_met = "True" in line
            elif "User input needed:" in line:
                user_input_needed = "True" in line

        feedback_msg = AIMessage(content=f"Evaluator Feedback: {feedback}")
        state["messages"].append(feedback_msg)
        state["feedback_on_work"] = feedback
        state["success_criteria_met"] = success_criteria_met
        state["user_input_needed"] = user_input_needed
        return state

    def format_conversation(self, messages: List[Any]) -> str:
        conv = ""
        for m in messages:
            if isinstance(m, HumanMessage):
                conv += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                conv += f"Assistant: {m.content or '[Tool use]'}\n"
        return conv

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        return "worker"

    # ---------- Graph Construction ----------
    async def build_graph(self):
        graph_builder = StateGraph(State)

        graph_builder.add_node("START", lambda state: state)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", self.tools_node)
        graph_builder.add_node("evaluator", self.evaluator)
        graph_builder.add_node("END", lambda state: state)

        graph_builder.add_edge("START", "worker")
        graph_builder.add_conditional_edges("worker", self.worker_router, {
            "tools": "tools",
            "evaluator": "evaluator"
        })
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {
            "worker": "worker",
            "END": "END"
        })

        # ✅ FIX: Set entry point before compiling
        graph_builder.set_entry_point("START")
        graph_builder.set_finish_point("END")   

        self.graph = graph_builder.compile(checkpointer=self.memory)

    # ---------- Public API ----------
    async def run_superstep(self, message: str, success_criteria: Optional[str], history: List[Dict[str, str]]):
        if self.graph is None:
            raise RuntimeError("Sidekick graph not built. Did you call setup()?")

        config = {"configurable": {"thread_id": self.sidekick_id}}

        state: State = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "subtasks": None,
        }

        result = await self.graph.ainvoke(state, config=config)

        assistant_reply_text = ""
        evaluator_feedback_text = ""
        for m in result["messages"]:
            if isinstance(m, AIMessage):
                if m.content and m.content.startswith("Evaluator Feedback:"):
                    evaluator_feedback_text = m.content
                else:
                    assistant_reply_text = m.content

        return (history or []) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": assistant_reply_text},
            {"role": "assistant", "content": evaluator_feedback_text},
        ]

    # ---------- Cleanup ----------
    def cleanup(self):
        async def _shutdown():
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_shutdown())
        except RuntimeError:
            asyncio.run(_shutdown())
        finally:
            self.browser = None
            self.playwright = None
