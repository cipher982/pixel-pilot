from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
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


class DualPathGraph:
    """Implements the dual-path architecture with terminal and visual paths."""

    RECURSION_LIMIT = 100

    def __init__(
        self, window_info: Optional[Dict[str, Any]], start_terminal: bool = False, llm_provider: str = "openai"
    ):
        """Initialize the dual-path system."""
        self.path_manager = PathManager()

        # Initialize shared state with window info
        self.path_manager.update_state({"window_info": window_info or {}})
        self.path_manager.switch_path("terminal" if start_terminal else "visual")

        # Initialize LLM
        self.llm = self._init_llm(llm_provider)

        # Initialize tools
        self.terminal_tool = TerminalTool()
        self.visual_ops = VisualOperations(window_info=window_info or {})

        # Build graphs
        self.terminal_graph = self._build_terminal_graph()
        self.visual_graph = self._build_visual_graph()

    def _init_llm(self, provider: str) -> Union[OpenAICompatibleChatModel, LocalTGIChatModel]:
        """Initialize LLM with structured output."""
        if provider == "local":
            llm = LocalTGIChatModel(base_url="http://localhost:8080")
        else:  # default to openai
            # llm = OpenAICompatibleChatModel(
            #     base_url="https://api.openai.com/v1", api_key=str(os.getenv("OPENAI_API_KEY")), model="gpt-4o-mini"
            # )
            llm = ChatOpenAI(model="gpt-4o-mini")
        return llm.with_structured_output(ActionResponse)  # type: ignore

    def _decide_next_action(self, state: SharedState) -> SharedState:
        """Use LLM to decide next action."""
        messages = [
            SystemMessage(
                content="""You are an AI assistant that decides what action to take next.
            Based on the task description and current state, determine the next action.
            You can use terminal commands or visual operations to accomplish the task.
            
            When the task is complete:
            1. Set is_task_complete=true
            2. Set next_path="end"
            3. Explain why the task is complete
            
            When switching modes:
            1. Set next_path="terminal" or "visual"
            2. Explain why switching is needed"""
            ),
            HumanMessage(content=self._create_decision_prompt(state)),
        ]

        response: ActionResponse = self.llm.invoke(messages)  # type: ignore
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
        """Create prompt for LLM decision making."""
        return f"""
        Task: {state['task_description']}
        Current Path: {state['current_path']}
        Last Output: {state['last_output']}
        Action History: {[a.model_dump() for a in state['action_history']]}
        
        Decide:
        1. Is the task complete? If yes, mark complete and end
        2. Do we need to switch modes? If yes, specify which mode
        3. What action should be taken next?
        """

    def _build_terminal_graph(self) -> CompiledStateGraph:
        """Build the terminal-focused operation path."""
        workflow = StateGraph(SharedState)

        # Add nodes for terminal operations
        workflow.add_node("decide_action", self._decide_next_action)
        workflow.add_node("execute_command", self.terminal_tool.execute_command)
        workflow.add_node("analyze_output", self.terminal_tool.analyze_output)

        # Set entry point and edges
        workflow.set_entry_point("decide_action")
        workflow.add_edge("decide_action", "execute_command")
        workflow.add_edge("execute_command", "analyze_output")

        # Add conditional edge for completion
        workflow.add_conditional_edges(
            "analyze_output", self._should_continue, {"continue": "decide_action", "end": END, "switch_to_visual": END}
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
                return {
                    "status": "completed",
                    "task_status": task_status,
                    "result": final_state.get("last_output", ""),
                    "context": final_state.get("context", {}),
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
        # Initialize state with task
        self.path_manager.update_state({"task_description": task_description, "task_status": "in_progress"})
        logger.info(f"Starting execution with task: {task_description}")

        while True:
            result = self._execute_path(self.path_manager.state["current_path"])
            logger.info(f"Path execution completed with status: {result.get('status')}")

            if result.get("status") != "continue":
                if result.get("task_status") == "completed":
                    logger.info("Task completed successfully")
                else:
                    logger.warning("Task did not complete as expected")
                return result
