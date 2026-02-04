"""Tests for the navigate_to dynamic routing feature (hybrid execution pattern).

Tests the navigate_to synthetic tool on EventLoopNode and the executor's
validation of navigation targets.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from framework.graph.edge import EdgeCondition, EdgeSpec, GraphSpec
from framework.graph.event_loop_node import (
    EventLoopNode,
    LoopConfig,
)
from framework.graph.node import NodeContext, NodeResult, NodeSpec, SharedMemory
from framework.llm.provider import LLMProvider, LLMResponse, Tool
from framework.llm.stream_events import (
    FinishEvent,
    TextDeltaEvent,
    ToolCallEvent,
)
from framework.runtime.core import Runtime

# ---------------------------------------------------------------------------
# Mock LLM (same pattern as test_event_loop_node.py)
# ---------------------------------------------------------------------------


class MockStreamingLLM(LLMProvider):
    """Mock LLM that yields pre-programmed StreamEvent sequences."""

    def __init__(self, scenarios: list[list] | None = None):
        self.scenarios = scenarios or []
        self._call_index = 0
        self.stream_calls: list[dict] = []

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[Tool] | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator:
        self.stream_calls.append({"messages": messages, "system": system, "tools": tools})
        if not self.scenarios:
            return
        events = self.scenarios[self._call_index % len(self.scenarios)]
        self._call_index += 1
        for event in events:
            yield event

    def complete(self, messages, system="", **kwargs) -> LLMResponse:
        return LLMResponse(content="Summary.", model="mock", stop_reason="stop")

    def complete_with_tools(self, messages, system, tools, tool_executor, **kwargs) -> LLMResponse:
        return LLMResponse(content="", model="mock", stop_reason="stop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text_scenario(text: str) -> list:
    return [
        TextDeltaEvent(content=text, snapshot=text),
        FinishEvent(stop_reason="stop", input_tokens=10, output_tokens=5, model="mock"),
    ]


def tool_call_scenario(tool_name: str, tool_input: dict, tool_use_id: str = "call_1") -> list:
    return [
        ToolCallEvent(tool_use_id=tool_use_id, tool_name=tool_name, tool_input=tool_input),
        FinishEvent(stop_reason="tool_calls", input_tokens=10, output_tokens=5, model="mock"),
    ]


@pytest.fixture
def runtime():
    rt = MagicMock(spec=Runtime)
    rt.start_run = MagicMock(return_value="run_1")
    rt.decide = MagicMock(return_value="dec_1")
    rt.record_outcome = MagicMock()
    rt.end_run = MagicMock()
    rt.report_problem = MagicMock()
    rt.set_node = MagicMock()
    return rt


@pytest.fixture
def memory():
    return SharedMemory()


def build_ctx(runtime, node_spec, memory, llm, tools=None, input_data=None):
    return NodeContext(
        runtime=runtime,
        node_id=node_spec.id,
        node_spec=node_spec,
        memory=memory,
        input_data=input_data or {},
        llm=llm,
        available_tools=tools or [],
    )


# ===========================================================================
# navigate_to tool availability
# ===========================================================================


class TestNavigateToToolAvailability:
    @pytest.mark.asyncio
    async def test_no_tool_without_targets(self, runtime, memory):
        """navigate_to tool should NOT appear when allowed_navigation_targets is empty."""
        spec = NodeSpec(
            id="node_a",
            name="Node A",
            description="test",
            node_type="event_loop",
            output_keys=["result"],
            allowed_navigation_targets=[],
        )
        llm = MockStreamingLLM(
            scenarios=[
                tool_call_scenario("set_output", {"key": "result", "value": "done"}),
                text_scenario("Done."),
            ]
        )
        ctx = build_ctx(runtime, spec, memory, llm)
        node = EventLoopNode(config=LoopConfig(max_iterations=5))
        await node.execute(ctx)

        assert llm.stream_calls, "LLM should have been called"
        tools_sent = llm.stream_calls[0]["tools"] or []
        tool_names = [t.name for t in tools_sent]
        assert "navigate_to" not in tool_names
        assert "set_output" in tool_names

    @pytest.mark.asyncio
    async def test_tool_present_with_targets(self, runtime, memory):
        """navigate_to tool should appear when allowed_navigation_targets is set."""
        spec = NodeSpec(
            id="node_a",
            name="Node A",
            description="test",
            node_type="event_loop",
            output_keys=["result"],
            allowed_navigation_targets=["node_b", "node_c"],
        )
        llm = MockStreamingLLM(
            scenarios=[
                tool_call_scenario("set_output", {"key": "result", "value": "done"}),
                text_scenario("Done."),
            ]
        )
        ctx = build_ctx(runtime, spec, memory, llm)
        node = EventLoopNode(config=LoopConfig(max_iterations=5))
        await node.execute(ctx)

        tools_sent = llm.stream_calls[0]["tools"] or []
        tool_names = [t.name for t in tools_sent]
        assert "navigate_to" in tool_names
        assert "set_output" in tool_names


# ===========================================================================
# navigate_to tool execution
# ===========================================================================


class TestNavigateToExecution:
    @pytest.mark.asyncio
    async def test_valid_target_returns_next_node(self, runtime, memory):
        """Calling navigate_to with a valid target should set NodeResult.next_node."""
        spec = NodeSpec(
            id="approval",
            name="Approval",
            description="test",
            node_type="event_loop",
            output_keys=["approved", "revise"],
            nullable_output_keys=["approved", "revise"],
            allowed_navigation_targets=["review", "campaign"],
        )
        llm = MockStreamingLLM(
            scenarios=[
                tool_call_scenario(
                    "navigate_to",
                    {"target": "review", "reason": "User wants to go back"},
                ),
            ]
        )
        ctx = build_ctx(runtime, spec, memory, llm)
        node = EventLoopNode(config=LoopConfig(max_iterations=5))
        result = await node.execute(ctx)

        assert result.success is True
        assert result.next_node == "review"
        assert "User wants to go back" in (result.route_reason or "")

    @pytest.mark.asyncio
    async def test_invalid_target_continues_loop(self, runtime, memory):
        """Calling navigate_to with an invalid target should error and continue."""
        spec = NodeSpec(
            id="approval",
            name="Approval",
            description="test",
            node_type="event_loop",
            output_keys=["result"],
            allowed_navigation_targets=["review"],
        )
        llm = MockStreamingLLM(
            scenarios=[
                # Turn 1: try invalid target
                tool_call_scenario(
                    "navigate_to",
                    {"target": "nonexistent", "reason": "test"},
                ),
                # Turn 2: set output normally
                tool_call_scenario("set_output", {"key": "result", "value": "ok"}),
                # Turn 3: text -> implicit judge accepts
                text_scenario("Done."),
            ]
        )
        ctx = build_ctx(runtime, spec, memory, llm)
        node = EventLoopNode(config=LoopConfig(max_iterations=5))
        result = await node.execute(ctx)

        assert result.success is True
        assert result.next_node is None
        assert result.output.get("result") == "ok"

    @pytest.mark.asyncio
    async def test_navigate_to_with_partial_outputs(self, runtime, memory):
        """Outputs set before navigate_to should still be in the result dict
        but the node exits via navigation, not via normal completion."""
        spec = NodeSpec(
            id="approval",
            name="Approval",
            description="test",
            node_type="event_loop",
            output_keys=["draft", "final"],
            nullable_output_keys=["draft", "final"],
            allowed_navigation_targets=["review"],
        )
        llm = MockStreamingLLM(
            scenarios=[
                # Turn 1: set one output
                tool_call_scenario(
                    "set_output",
                    {"key": "draft", "value": "v1"},
                    tool_use_id="call_1",
                ),
                # Turn 2: navigate away
                tool_call_scenario(
                    "navigate_to",
                    {"target": "review", "reason": "go back"},
                    tool_use_id="call_2",
                ),
            ]
        )
        ctx = build_ctx(runtime, spec, memory, llm)
        node = EventLoopNode(config=LoopConfig(max_iterations=5))
        result = await node.execute(ctx)

        assert result.success is True
        assert result.next_node == "review"
        assert result.output.get("draft") == "v1"


# ===========================================================================
# navigate_to handler unit tests
# ===========================================================================


class TestNavigateToHandler:
    def test_valid_target(self):
        """_handle_navigate_to should succeed for valid targets."""
        node = EventLoopNode()
        result = node._handle_navigate_to(
            {"target": "review", "reason": "go back"},
            allowed_targets=["review", "campaign"],
        )
        assert not result.is_error
        assert "review" in result.content

    def test_invalid_target(self):
        """_handle_navigate_to should error for invalid targets."""
        node = EventLoopNode()
        result = node._handle_navigate_to(
            {"target": "nonexistent", "reason": "test"},
            allowed_targets=["review", "campaign"],
        )
        assert result.is_error
        assert "Invalid" in result.content

    def test_empty_target(self):
        """_handle_navigate_to should error when target is empty."""
        node = EventLoopNode()
        result = node._handle_navigate_to(
            {"target": "", "reason": "test"},
            allowed_targets=["review"],
        )
        assert result.is_error


# ===========================================================================
# navigate_to tool builder
# ===========================================================================


class TestNavigateToToolBuilder:
    def test_no_tool_with_empty_targets(self):
        """Should return None when no targets allowed."""
        node = EventLoopNode()
        tool = node._build_navigate_to_tool([])
        assert tool is None

    def test_tool_with_targets(self):
        """Should return a Tool with enum of allowed targets."""
        node = EventLoopNode()
        tool = node._build_navigate_to_tool(["review", "campaign"])
        assert tool is not None
        assert tool.name == "navigate_to"
        assert tool.parameters["properties"]["target"]["enum"] == ["review", "campaign"]
        assert "required" in tool.parameters
        assert "target" in tool.parameters["required"]
        assert "reason" in tool.parameters["required"]


# ===========================================================================
# GraphSpec validation for navigation targets
# ===========================================================================


class TestGraphSpecNavValidation:
    def test_valid_targets_pass(self):
        """Navigation targets that reference existing nodes should pass validation."""
        graph = GraphSpec(
            id="test",
            goal_id="test_goal",
            entry_node="a",
            nodes=[
                NodeSpec(
                    id="a",
                    name="A",
                    description="t",
                    node_type="event_loop",
                    allowed_navigation_targets=["b"],
                ),
                NodeSpec(id="b", name="B", description="t", node_type="event_loop"),
            ],
            edges=[
                EdgeSpec(id="a_to_b", source="a", target="b"),
            ],
            terminal_nodes=["b"],
        )
        errors = graph.validate()
        nav_errors = [e for e in errors if "allowed_navigation_target" in e]
        assert nav_errors == []

    def test_invalid_targets_fail(self):
        """Navigation targets referencing non-existent nodes should fail."""
        graph = GraphSpec(
            id="test",
            goal_id="test_goal",
            entry_node="a",
            nodes=[
                NodeSpec(
                    id="a",
                    name="A",
                    description="t",
                    node_type="event_loop",
                    allowed_navigation_targets=["nonexistent"],
                ),
                NodeSpec(id="b", name="B", description="t", node_type="event_loop"),
            ],
            edges=[
                EdgeSpec(id="a_to_b", source="a", target="b"),
            ],
            terminal_nodes=["b"],
        )
        errors = graph.validate()
        nav_errors = [e for e in errors if "allowed_navigation_target" in e]
        assert len(nav_errors) == 1
        assert "nonexistent" in nav_errors[0]

    def test_empty_targets_pass(self):
        """Nodes with no navigation targets should pass (backward compatible)."""
        graph = GraphSpec(
            id="test",
            goal_id="test_goal",
            entry_node="a",
            nodes=[
                NodeSpec(id="a", name="A", description="t", node_type="event_loop"),
                NodeSpec(id="b", name="B", description="t", node_type="event_loop"),
            ],
            edges=[
                EdgeSpec(id="a_to_b", source="a", target="b"),
            ],
            terminal_nodes=["b"],
        )
        errors = graph.validate()
        nav_errors = [e for e in errors if "allowed_navigation_target" in e]
        assert nav_errors == []


# ===========================================================================
# Executor integration: navigate_to routing
# ===========================================================================


class TestExecutorNavigation:
    """Tests for executor handling of navigate_to results."""

    def _make_graph(self, a_targets: list[str] | None = None) -> GraphSpec:
        """Build a simple A -> B -> C graph where A can navigate."""
        return GraphSpec(
            id="test",
            goal_id="test_goal",
            entry_node="a",
            nodes=[
                NodeSpec(
                    id="a",
                    name="A",
                    description="first",
                    node_type="event_loop",
                    output_keys=["out"],
                    allowed_navigation_targets=a_targets or [],
                    max_node_visits=3,
                ),
                NodeSpec(
                    id="b",
                    name="B",
                    description="second",
                    node_type="event_loop",
                    output_keys=["out"],
                    max_node_visits=3,
                ),
                NodeSpec(
                    id="c",
                    name="C",
                    description="third",
                    node_type="event_loop",
                    output_keys=["out"],
                ),
            ],
            edges=[
                EdgeSpec(
                    id="a_to_b",
                    source="a",
                    target="b",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
                EdgeSpec(
                    id="b_to_c",
                    source="b",
                    target="c",
                    condition=EdgeCondition.ON_SUCCESS,
                ),
            ],
            terminal_nodes=["c"],
        )

    def _make_runtime(self):
        rt = MagicMock(spec=Runtime)
        rt.start_run = MagicMock(return_value="run_1")
        rt.decide = MagicMock(return_value="dec_1")
        rt.record_outcome = MagicMock()
        rt.end_run = MagicMock()
        rt.report_problem = MagicMock()
        rt.set_node = MagicMock()
        return rt

    @pytest.mark.asyncio
    async def test_navigation_changes_next_node(self):
        """When a node returns next_node in allowed_navigation_targets,
        executor should route to it."""
        from framework.graph.executor import GraphExecutor

        graph = self._make_graph(a_targets=["b", "c"])
        call_count = {"a": 0, "b": 0, "c": 0}

        class MockNode:
            def __init__(self, node_id, next_target=None):
                self.node_id = node_id
                self.next_target = next_target

            def validate_input(self, ctx):
                return []

            async def execute(self, ctx):
                call_count[self.node_id] += 1
                if self.next_target and call_count[self.node_id] == 1:
                    return NodeResult(
                        success=True,
                        output={"out": "navigated"},
                        next_node=self.next_target,
                        route_reason="user requested",
                    )
                return NodeResult(success=True, output={"out": f"done_{self.node_id}"})

        rt = self._make_runtime()
        executor = GraphExecutor(
            runtime=rt,
            node_registry={
                "a": MockNode("a", next_target="c"),  # Skip B, go to C
                "b": MockNode("b"),
                "c": MockNode("c"),
            },
        )

        _result = await executor.execute(
            graph=graph,
            goal=MagicMock(name="test_goal", description="test", success_criteria="test"),
        )

        # A should navigate directly to C, skipping B
        assert call_count["a"] == 1
        assert call_count["b"] == 0
        assert call_count["c"] == 1

    @pytest.mark.asyncio
    async def test_unauthorized_navigation_falls_through(self):
        """When next_node is not in allowed_navigation_targets,
        executor should fall through to normal edge evaluation."""
        from framework.graph.executor import GraphExecutor

        # A can only navigate to B, not C
        graph = self._make_graph(a_targets=["b"])
        call_count = {"a": 0, "b": 0, "c": 0}

        class MockNode:
            def __init__(self, node_id):
                self.node_id = node_id

            def validate_input(self, ctx):
                return []

            async def execute(self, ctx):
                call_count[self.node_id] += 1
                if self.node_id == "a":
                    # Try to navigate to C (not allowed)
                    return NodeResult(
                        success=True,
                        output={"out": "try_c"},
                        next_node="c",
                    )
                return NodeResult(success=True, output={"out": f"done_{self.node_id}"})

        rt = self._make_runtime()
        executor = GraphExecutor(
            runtime=rt,
            node_registry={
                "a": MockNode("a"),
                "b": MockNode("b"),
                "c": MockNode("c"),
            },
        )

        _result = await executor.execute(
            graph=graph,
            goal=MagicMock(name="test_goal", description="test", success_criteria="test"),
        )

        # A's navigation to C should be blocked; normal edge A->B fires
        assert call_count["a"] == 1
        assert call_count["b"] == 1
        assert call_count["c"] == 1  # B->C edge fires normally
