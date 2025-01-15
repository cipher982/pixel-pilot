import os
import platform
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Set
from typing import Union
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from pixelpilot.llms.openai_wrapper import OpenAICompatibleChatModel
from pixelpilot.llms.tgi_wrapper import LocalTGIChatModel
from pixelpilot.logger import setup_logger
from pixelpilot.models import ActionResponse
from pixelpilot.state_management import PathManager
from pixelpilot.state_management import SharedState
from pixelpilot.tools.terminal import TerminalTool
from pixelpilot.visual_ops import VisualOperations

logger = setup_logger(__name__)


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_LOCAL_URL = "http://localhost:8080"


class MetadataTracker:
    """Tracks execution metadata"""

    def __init__(self):
        self.start_time = datetime.now()
        self.path_transitions: List[str] = []
        self.models_used: Set[str] = set()
        self.token_usage = {"decision": 0, "summary": 0}
        self.confidence = 0.0

    def add_path_transition(self, path: str) -> None:
        self.path_transitions.append(path)

    def update_token_usage(self, model_type: str, tokens: int) -> None:
        self.token_usage[model_type] += tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": datetime.now(),
            "path_transitions": self.path_transitions,
            "models_used": list(self.models_used),
            "token_usage": self.token_usage,
            "confidence": self.confidence,
        }


class DualPathGraph:
    """Implements the dual-path architecture with terminal and visual paths."""

    RECURSION_LIMIT = 100

    def __init__(
        self, window_info: Optional[Dict[str, Any]], start_terminal: bool = False, llm_provider: str = "openai"
    ):
        """Initialize the dual-path system."""
        self.path_manager = PathManager()
        self.metadata = MetadataTracker()

        # Initialize system info
        system_info = {
            "os_type": platform.system().lower(),
            "os_version": platform.release(),
            "shell": os.environ.get("SHELL", "unknown"),
            "arch": platform.machine(),
            "python_version": platform.python_version(),
        }

        # Initialize shared state with window info and system info
        self.path_manager.update_state({"window_info": window_info or {}, "context": {"system_info": system_info}})
        initial_path = "terminal" if start_terminal else "visual"
        self.path_manager.switch_path(initial_path)
        self.metadata.add_path_transition(initial_path)

        # Initialize LLMs for different purposes
        self.decision_llm = self._init_decision_llm(llm_provider)
        self.summary_llm = self._init_summary_llm(llm_provider)
        self.metadata.models_used.add(self._get_model_name(llm_provider))

        # Initialize tools
        self.terminal_tool = TerminalTool()
        self.visual_ops = VisualOperations(window_info=window_info or {})

        # Build graphs
        self.terminal_graph = self._build_terminal_graph()
        self.visual_graph = self._build_visual_graph()

    def _get_model_name(self, provider: str) -> str:
        return "gpt-4o" if provider == "openai" else "local-model"

    def _init_decision_llm(self, provider: str) -> Union[OpenAICompatibleChatModel, LocalTGIChatModel]:
        """Initialize LLM for structured decision making."""
        if provider == "local":
            llm = LocalTGIChatModel(base_url=DEFAULT_LOCAL_URL)
        else:  # default to openai
            llm = ChatOpenAI(model=DEFAULT_OPENAI_MODEL)
        return llm.with_structured_output(ActionResponse)  # type: ignore

    def _init_summary_llm(self, provider: str) -> ChatOpenAI:
        """Initialize LLM for free-form text generation."""
        if provider == "local":
            return ChatOpenAI(model=DEFAULT_OPENAI_MODEL)  # Fallback to OpenAI for summaries
        return ChatOpenAI(model=DEFAULT_OPENAI_MODEL)

    def _decide_next_action(self, state: SharedState) -> SharedState:
        """Use LLM to decide next action."""
        messages = [
            SystemMessage(content=self._get_decision_system_prompt()),
            HumanMessage(content=self._create_decision_prompt(state)),
        ]

        # Log the prompt for debugging
        logger.debug(f"Sending prompt to LLM:\n{messages[1].content}")

        response: ActionResponse = self.decision_llm.invoke(messages)  # type: ignore

        # Log the raw response
        logger.debug(f"LLM Response: {response.model_dump_json(indent=2)}")

        # Update metadata
        self.metadata.confidence = max(self.metadata.confidence, response.confidence)
        if response.next_path != state["current_path"]:
            self.metadata.add_path_transition(response.next_path)

        # Update state
        state["context"]["next_action"] = response.action.model_dump()
        state["context"]["next_path"] = response.next_path
        state["context"]["reasoning"] = response.reasoning

        # Update task status if complete
        if response.is_task_complete:
            state["task_status"] = "completed"
            state["context"]["next_path"] = "end"
            logger.info(f"Task marked as complete. Reason: {response.reasoning}")

        return state

    def _create_decision_prompt(self, state: SharedState) -> str:
        """Create prompt for LLM decision making with comprehensive system context."""
        command_history = [f"Command: {cmd}" for cmd in state.get("command_history", [])]
        last_result = state["context"].get("last_action_result", {})

        return f"""
Current Task Information:
    Description: {state['task_description']}
    Status: {state.get('task_status', 'unknown')}
    Current Path: {state['current_path']}

System Context:
    OS: {state["context"].get("system_info", {}).get("os_type", "unknown")}
    Shell: {state["context"].get("system_info", {}).get("shell", "unknown")}
    Working Directory: {state.get("current_directory", "unknown")}

Last Action Result:
    Success: {last_result.get("success", False)}
    Output: {state.get("last_output", "No output")}
    Error: {last_result.get("error", "None")}

Command History:
{'\n'.join(command_history)}

Decide and respond with:
1. action: The next action to take, including:
   - type: "terminal" or "visual"
   - command: The command string (e.g., "df -h /")
   - args: Optional dictionary of subprocess arguments (e.g., {{"cwd": "/tmp"}}) or null
2. next_path: 'terminal', 'visual', or 'end'
3. is_task_complete: true/false
4. reasoning: Why you made this decision
5. confidence: Float between 0-1 indicating your confidence level
"""

    def _build_terminal_graph(self) -> CompiledStateGraph:
        """Build the terminal-focused operation path."""
        workflow = StateGraph(SharedState)

        # Add nodes for terminal operations
        workflow.add_node("decide_action", self._decide_next_action)
        workflow.add_node("execute_command", self.terminal_tool.execute_command)
        workflow.add_node("analyze_output", self.terminal_tool.analyze_output)
        workflow.add_node("summarize", self.summarize_result)  # Add summarization node

        # Set entry point and edges
        workflow.set_entry_point("decide_action")
        workflow.add_edge("decide_action", "execute_command")
        workflow.add_edge("execute_command", "analyze_output")
        workflow.add_edge("analyze_output", "summarize")  # Add edge to summarization

        # Add conditional edge for completion
        workflow.add_conditional_edges(
            "summarize", self._should_continue, {"continue": "decide_action", "end": END, "switch_to_visual": END}
        )

        return workflow.compile()

    def _build_visual_graph(self) -> CompiledStateGraph:
        """Build the vision-focused operation path."""
        workflow = StateGraph(SharedState)

        # Add nodes for visual operations
        workflow.add_node("decide_action", self._decide_next_action)
        workflow.add_node("execute_visual", self.visual_ops.execute_visual_action)
        workflow.add_node("analyze_result", self.visual_ops.analyze_visual_result)

        # Set entry point and edges
        workflow.set_entry_point("decide_action")
        workflow.add_edge("decide_action", "execute_visual")
        workflow.add_edge("execute_visual", "analyze_result")

        # Add conditional edge for completion
        workflow.add_conditional_edges(
            "analyze_result",
            self._should_continue,
            {"continue": "decide_action", "end": END, "switch_to_terminal": END},
        )

        return workflow.compile()

    def _should_continue(self, state: SharedState) -> str:
        """Determine if we should continue or switch paths."""
        next_path = state["context"].get("next_path")
        task_status = state.get("task_status")

        if task_status == "completed":
            return "end"
        elif next_path == "terminal" and state["current_path"] == "visual":
            return "switch_to_terminal"
        elif next_path == "visual" and state["current_path"] == "terminal":
            return "switch_to_visual"

        return "continue"

    def _execute_path(self, path: str) -> Dict[str, Any]:
        """Execute a single path and handle switching."""
        path = cast(Literal["terminal", "visual"], path)
        logger.info(f"Executing path: {path}")

        graph = self.terminal_graph if path == "terminal" else self.visual_graph
        state = self.path_manager.get_state()

        try:
            final_state = graph.invoke(state, {"recursion_limit": self.RECURSION_LIMIT})
            task_status = final_state.get("task_status")

            # Check for completion first
            if task_status == "completed":
                # Log final state for debugging
                logger.debug(f"Final state on completion: {final_state}")
                return {
                    "status": "completed",
                    "task_status": task_status,
                    "result": final_state.get("last_output", ""),
                    "context": final_state.get("context", {}),
                    "summary": final_state.get("context", {}).get("summary"),  # Include summary in result
                }

            # Handle path switching
            next_path = final_state["context"].get("next_path")
            if next_path in ["terminal", "visual"]:
                self.path_manager.switch_path(cast(Literal["terminal", "visual"], next_path))
                return {"status": "continue"}

            # Default case
            return {
                "status": "failed",
                "task_status": task_status,
                "error": "Task not completed and no path switch requested",
                "context": final_state.get("context", {}),
            }

        except Exception as e:
            logger.error(f"Path execution failed: {e}")
            return {"status": "failed", "error": str(e)}

    def run(self, task_description: str) -> Dict[str, Any]:
        """Run the dual-path system, switching between paths as needed."""
        # Initialize state with task and metadata
        self.path_manager.update_state(
            {
                "task_description": task_description,
                "task_status": "in_progress",
                "context": {"start_time": self.metadata.start_time},
            }
        )
        logger.info(f"Starting execution with task: {task_description}")

        while True:
            result = self._execute_path(self.path_manager.state["current_path"])
            logger.info(f"Path execution completed with status: {result.get('status')}")

            if result.get("status") != "continue":
                # Update final metadata
                self.path_manager.update_state({"context": self.metadata.to_dict()})
                if result.get("task_status") == "completed":
                    logger.info("Task completed successfully")
                else:
                    logger.warning("Task did not complete as expected")
                return result

    def summarize_result(self, state: SharedState) -> SharedState:
        """Generate a user-friendly summary of the task result."""
        # Log state for debugging
        logger.debug("Summarizing result. State contains:")
        logger.debug(f"- task_description: {state['task_description']}")
        logger.debug(f"- task_status: {state['task_status']}")
        logger.debug(f"- last_action_result: {state['context'].get('last_action_result')}")
        logger.debug(f"- last_output: {state.get('last_output')}")

        messages = [
            SystemMessage(
                content="""You are an AI assistant that summarizes task results in a clear, concise way.
                Based on the task description and output, create a user-friendly summary.
                Focus on the key information the user needs to know.
                Be direct and to the point."""
            ),
            HumanMessage(
                content=f"""
                Task: {state['task_description']}
                Raw Output: {state['context'].get('last_action_result', {}).get('output', 'No output available')}
                
                Provide a clear, direct summary focusing on the key information.
                """
            ),
        ]

        response = self.summary_llm.invoke(messages)
        logger.info(f"Summary generated: {response.content}")
        state["context"]["summary"] = response.content

        # Also update task status since this is our final node
        if state.get("task_status") == "completed":
            state["context"]["next_path"] = "end"

        return state

    def _get_decision_system_prompt(self) -> str:
        """Get the system prompt for decision making."""
        return """You are an AI assistant that decides what action to take next.
        Based on the task description, current state, and command history, determine the next action.
        
        You should:
        1. Review the command history to understand what has been done
        2. Check the last output to see if it was successful
        3. Plan the next action based on progress so far
        4. Track overall task completion
        
        When the task is complete:
        1. Set is_task_complete=true
        2. Set next_path="end"
        3. Explain why the task is complete
        4. Set confidence based on certainty of completion
        
        When handling errors:
        1. Analyze the error message
        2. Check if the error is because the action was already completed
        3. Move on to the next required action
        
        Always set confidence between 0 and 1 as a float."""
