"""
Chapter 16: Agent Debugging Tools
================================

Implements debugging utilities for AI agents:
- Step-through execution
- State inspection
- Replay and time-travel debugging
- Breakpoints and watchpoints
- Conversation history analysis

Essential tools for understanding and fixing agent behavior.
"""

import asyncio
import time
import json
import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import deque


class ExecutionState(Enum):
    """State of agent execution."""
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    STOPPED = "stopped"
    COMPLETED = "completed"


class BreakpointType(Enum):
    """Types of breakpoints."""
    STEP = "step"           # Break at every step
    TOOL = "tool"           # Break on tool calls
    LLM = "llm"             # Break on LLM calls
    CONDITION = "condition" # Break when condition is met
    ERROR = "error"         # Break on errors


@dataclass
class Breakpoint:
    """A debugging breakpoint."""
    id: str
    type: BreakpointType
    condition: Optional[str] = None  # For conditional breakpoints
    tool_name: Optional[str] = None  # For tool breakpoints
    enabled: bool = True
    hit_count: int = 0

    def should_break(self, context: dict) -> bool:
        """Check if this breakpoint should trigger."""
        if not self.enabled:
            return False

        if self.type == BreakpointType.STEP:
            return True

        if self.type == BreakpointType.TOOL:
            return context.get("tool_name") == self.tool_name

        if self.type == BreakpointType.LLM:
            return context.get("step_type") == "llm"

        if self.type == BreakpointType.CONDITION:
            try:
                return eval(self.condition, {"context": context})
            except Exception:
                return False

        if self.type == BreakpointType.ERROR:
            return context.get("error") is not None

        return False


@dataclass
class DebugSnapshot:
    """A snapshot of agent state at a point in time."""
    step_number: int
    timestamp: float
    state: dict
    input_data: Any
    output_data: Any
    step_type: str
    tool_name: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "step_number": self.step_number,
            "timestamp": self.timestamp,
            "step_type": self.step_type,
            "tool_name": self.tool_name,
            "input": str(self.input_data)[:200],
            "output": str(self.output_data)[:200],
            "error": self.error,
            "state_keys": list(self.state.keys())
        }


@dataclass
class Watchpoint:
    """Watch a state variable for changes."""
    id: str
    variable_path: str  # Dot-separated path like "state.memory.context"
    previous_value: Any = None
    change_count: int = 0
    break_on_change: bool = True

    def check(self, state: dict) -> tuple[bool, Any]:
        """Check if watched variable changed. Returns (changed, new_value)."""
        value = self._get_value(state)
        changed = value != self.previous_value
        if changed:
            self.change_count += 1
            self.previous_value = copy.deepcopy(value)
        return changed, value

    def _get_value(self, state: dict) -> Any:
        """Get value at variable path."""
        parts = self.variable_path.split(".")
        current = state
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current


class AgentDebugger:
    """
    Interactive debugger for AI agents.
    Provides step-through execution, breakpoints, and state inspection.
    """

    def __init__(self, max_history: int = 1000):
        self.execution_state = ExecutionState.STOPPED
        self.breakpoints: dict[str, Breakpoint] = {}
        self.watchpoints: dict[str, Watchpoint] = {}
        self.snapshots: deque[DebugSnapshot] = deque(maxlen=max_history)
        self.current_step = 0
        self.step_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Start unpaused
        self._step_callbacks: list[Callable] = []
        self._breakpoint_callbacks: list[Callable] = []

    # ==========================================================================
    # Breakpoint Management
    # ==========================================================================

    def add_breakpoint(self, bp: Breakpoint) -> str:
        """Add a breakpoint."""
        self.breakpoints[bp.id] = bp
        return bp.id

    def remove_breakpoint(self, bp_id: str) -> bool:
        """Remove a breakpoint."""
        if bp_id in self.breakpoints:
            del self.breakpoints[bp_id]
            return True
        return False

    def enable_breakpoint(self, bp_id: str):
        """Enable a breakpoint."""
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].enabled = True

    def disable_breakpoint(self, bp_id: str):
        """Disable a breakpoint."""
        if bp_id in self.breakpoints:
            self.breakpoints[bp_id].enabled = False

    def add_tool_breakpoint(self, tool_name: str) -> str:
        """Convenience: add a breakpoint for a specific tool."""
        bp = Breakpoint(
            id=f"tool_{tool_name}",
            type=BreakpointType.TOOL,
            tool_name=tool_name
        )
        return self.add_breakpoint(bp)

    def add_conditional_breakpoint(self, condition: str, bp_id: Optional[str] = None) -> str:
        """Add a conditional breakpoint."""
        bp = Breakpoint(
            id=bp_id or f"cond_{len(self.breakpoints)}",
            type=BreakpointType.CONDITION,
            condition=condition
        )
        return self.add_breakpoint(bp)

    # ==========================================================================
    # Watchpoint Management
    # ==========================================================================

    def add_watchpoint(self, variable_path: str, break_on_change: bool = True) -> str:
        """Add a watchpoint for a state variable."""
        wp_id = f"watch_{variable_path.replace('.', '_')}"
        self.watchpoints[wp_id] = Watchpoint(
            id=wp_id,
            variable_path=variable_path,
            break_on_change=break_on_change
        )
        return wp_id

    def remove_watchpoint(self, wp_id: str) -> bool:
        """Remove a watchpoint."""
        if wp_id in self.watchpoints:
            del self.watchpoints[wp_id]
            return True
        return False

    # ==========================================================================
    # Execution Control
    # ==========================================================================

    async def pause(self):
        """Pause execution at next step."""
        self.execution_state = ExecutionState.PAUSED
        self._pause_event.clear()

    async def resume(self):
        """Resume execution."""
        self.execution_state = ExecutionState.RUNNING
        self._pause_event.set()

    async def step(self):
        """Execute a single step then pause."""
        self.execution_state = ExecutionState.STEPPING
        self._pause_event.set()

    async def stop(self):
        """Stop execution."""
        self.execution_state = ExecutionState.STOPPED
        self._pause_event.set()

    async def wait_for_continue(self):
        """Wait for continue signal (used by agent)."""
        await self._pause_event.wait()

        if self.execution_state == ExecutionState.STEPPING:
            self.execution_state = ExecutionState.PAUSED
            self._pause_event.clear()

    # ==========================================================================
    # Debug Step Processing
    # ==========================================================================

    async def record_step(self,
                           state: dict,
                           input_data: Any,
                           output_data: Any,
                           step_type: str,
                           tool_name: Optional[str] = None,
                           error: Optional[str] = None) -> bool:
        """
        Record a debug step. Returns True if a breakpoint was hit.

        Args:
            state: Current agent state
            input_data: Input to this step
            output_data: Output from this step
            step_type: Type of step (llm, tool, etc.)
            tool_name: Name of tool if applicable
            error: Error message if step failed

        Returns:
            True if execution should pause (breakpoint hit)
        """
        self.current_step += 1

        # Create snapshot
        snapshot = DebugSnapshot(
            step_number=self.current_step,
            timestamp=time.time(),
            state=copy.deepcopy(state),
            input_data=copy.deepcopy(input_data),
            output_data=copy.deepcopy(output_data),
            step_type=step_type,
            tool_name=tool_name,
            error=error
        )
        self.snapshots.append(snapshot)

        # Notify callbacks
        for callback in self._step_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(snapshot)
                else:
                    callback(snapshot)
            except Exception:
                pass

        # Build context for breakpoint evaluation
        context = {
            "step_number": self.current_step,
            "step_type": step_type,
            "tool_name": tool_name,
            "error": error,
            "state": state,
            "input": input_data,
            "output": output_data
        }

        # Check breakpoints
        should_break = False
        for bp in self.breakpoints.values():
            if bp.should_break(context):
                bp.hit_count += 1
                should_break = True
                for callback in self._breakpoint_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(bp, snapshot)
                        else:
                            callback(bp, snapshot)
                    except Exception:
                        pass

        # Check watchpoints
        for wp in self.watchpoints.values():
            changed, new_value = wp.check(state)
            if changed and wp.break_on_change:
                should_break = True
                print(f"[WATCH] {wp.variable_path} changed to: {new_value}")

        if should_break:
            self.execution_state = ExecutionState.PAUSED
            self._pause_event.clear()

        return should_break

    # ==========================================================================
    # State Inspection
    # ==========================================================================

    def get_current_snapshot(self) -> Optional[DebugSnapshot]:
        """Get the most recent snapshot."""
        return self.snapshots[-1] if self.snapshots else None

    def get_snapshot(self, step_number: int) -> Optional[DebugSnapshot]:
        """Get snapshot at a specific step."""
        for snapshot in self.snapshots:
            if snapshot.step_number == step_number:
                return snapshot
        return None

    def get_history(self, limit: int = 10) -> list[DebugSnapshot]:
        """Get recent execution history."""
        return list(self.snapshots)[-limit:]

    def inspect_state(self, path: Optional[str] = None) -> Any:
        """Inspect current state or a specific path."""
        snapshot = self.get_current_snapshot()
        if not snapshot:
            return None

        if not path:
            return snapshot.state

        # Navigate path
        parts = path.split(".")
        current = snapshot.state
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
        return current

    def diff_snapshots(self, step1: int, step2: int) -> dict:
        """Get differences between two snapshots."""
        snap1 = self.get_snapshot(step1)
        snap2 = self.get_snapshot(step2)

        if not snap1 or not snap2:
            return {"error": "Snapshot not found"}

        def deep_diff(d1: dict, d2: dict, path: str = "") -> list:
            diffs = []
            all_keys = set(d1.keys()) | set(d2.keys())

            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if key not in d1:
                    diffs.append({"path": key_path, "change": "added", "value": v2})
                elif key not in d2:
                    diffs.append({"path": key_path, "change": "removed", "value": v1})
                elif v1 != v2:
                    if isinstance(v1, dict) and isinstance(v2, dict):
                        diffs.extend(deep_diff(v1, v2, key_path))
                    else:
                        diffs.append({
                            "path": key_path,
                            "change": "modified",
                            "old": v1,
                            "new": v2
                        })

            return diffs

        return {
            "from_step": step1,
            "to_step": step2,
            "differences": deep_diff(snap1.state, snap2.state)
        }

    # ==========================================================================
    # Replay and Time-Travel
    # ==========================================================================

    def get_replay_data(self, from_step: int = 1, to_step: Optional[int] = None) -> list[dict]:
        """Get data needed to replay execution."""
        to_step = to_step or self.current_step
        replay_data = []

        for snapshot in self.snapshots:
            if from_step <= snapshot.step_number <= to_step:
                replay_data.append({
                    "step": snapshot.step_number,
                    "type": snapshot.step_type,
                    "tool": snapshot.tool_name,
                    "input": snapshot.input_data,
                    "output": snapshot.output_data,
                    "error": snapshot.error
                })

        return replay_data

    async def replay_to_step(self,
                              target_step: int,
                              replay_func: Callable) -> dict:
        """
        Replay execution up to a specific step.

        Args:
            target_step: Step to replay to
            replay_func: Function to replay each step

        Returns:
            State at target step
        """
        replay_data = self.get_replay_data(1, target_step)

        state = {}
        for step_data in replay_data:
            if asyncio.iscoroutinefunction(replay_func):
                state = await replay_func(state, step_data)
            else:
                state = replay_func(state, step_data)

        return state

    # ==========================================================================
    # Callbacks
    # ==========================================================================

    def on_step(self, callback: Callable):
        """Register callback for each step."""
        self._step_callbacks.append(callback)

    def on_breakpoint(self, callback: Callable):
        """Register callback when breakpoint is hit."""
        self._breakpoint_callbacks.append(callback)

    # ==========================================================================
    # Analysis
    # ==========================================================================

    def analyze_execution(self) -> dict:
        """Analyze execution patterns."""
        if not self.snapshots:
            return {"error": "No execution data"}

        step_types = {}
        tool_usage = {}
        errors = []
        durations = []

        prev_time = None
        for snapshot in self.snapshots:
            # Count step types
            step_types[snapshot.step_type] = step_types.get(snapshot.step_type, 0) + 1

            # Count tool usage
            if snapshot.tool_name:
                tool_usage[snapshot.tool_name] = tool_usage.get(snapshot.tool_name, 0) + 1

            # Track errors
            if snapshot.error:
                errors.append({
                    "step": snapshot.step_number,
                    "error": snapshot.error
                })

            # Track timing
            if prev_time:
                durations.append(snapshot.timestamp - prev_time)
            prev_time = snapshot.timestamp

        return {
            "total_steps": len(self.snapshots),
            "step_types": step_types,
            "tool_usage": tool_usage,
            "error_count": len(errors),
            "errors": errors[:10],  # First 10 errors
            "avg_step_duration": sum(durations) / len(durations) if durations else 0,
            "total_duration": self.snapshots[-1].timestamp - self.snapshots[0].timestamp
        }

    def export_session(self) -> str:
        """Export debug session to JSON."""
        return json.dumps({
            "snapshots": [s.to_dict() for s in self.snapshots],
            "breakpoints": [
                {"id": bp.id, "type": bp.type.value, "hit_count": bp.hit_count}
                for bp in self.breakpoints.values()
            ],
            "watchpoints": [
                {"id": wp.id, "path": wp.variable_path, "changes": wp.change_count}
                for wp in self.watchpoints.values()
            ],
            "analysis": self.analyze_execution()
        }, indent=2)


class ConversationDebugger:
    """
    Specialized debugger for analyzing agent conversations.
    """

    def __init__(self):
        self.turns: list[dict] = []
        self.message_graph: dict[str, list[str]] = {}  # message_id -> references

    def add_turn(self,
                 role: str,
                 content: str,
                 metadata: Optional[dict] = None) -> str:
        """Add a conversation turn."""
        turn_id = f"turn_{len(self.turns)}"
        self.turns.append({
            "id": turn_id,
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        return turn_id

    def analyze_conversation(self) -> dict:
        """Analyze conversation patterns."""
        if not self.turns:
            return {"error": "No conversation data"}

        role_counts = {}
        content_lengths = {"user": [], "assistant": [], "system": [], "tool": []}

        for turn in self.turns:
            role = turn["role"]
            role_counts[role] = role_counts.get(role, 0) + 1
            if role in content_lengths:
                content_lengths[role].append(len(turn["content"]))

        return {
            "total_turns": len(self.turns),
            "by_role": role_counts,
            "avg_length_by_role": {
                role: sum(lengths) / len(lengths) if lengths else 0
                for role, lengths in content_lengths.items()
            },
            "conversation_duration": (
                self.turns[-1]["timestamp"] - self.turns[0]["timestamp"]
                if len(self.turns) > 1 else 0
            )
        }

    def find_turns(self,
                   role: Optional[str] = None,
                   contains: Optional[str] = None) -> list[dict]:
        """Find turns matching criteria."""
        results = self.turns

        if role:
            results = [t for t in results if t["role"] == role]

        if contains:
            results = [t for t in results if contains.lower() in t["content"].lower()]

        return results

    def get_context_window(self, turn_index: int, window_size: int = 5) -> list[dict]:
        """Get conversation context around a specific turn."""
        start = max(0, turn_index - window_size)
        end = min(len(self.turns), turn_index + window_size + 1)
        return self.turns[start:end]


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of agent debugging tools."""
    print("=" * 60)
    print("Agent Debugging Tools Demonstration")
    print("=" * 60)

    # Create debugger
    debugger = AgentDebugger()

    # Set up breakpoints
    debugger.add_breakpoint(Breakpoint(
        id="bp_error",
        type=BreakpointType.ERROR
    ))

    debugger.add_tool_breakpoint("search")

    debugger.add_conditional_breakpoint(
        "context['step_number'] == 5",
        "bp_step_5"
    )

    # Set up watchpoint
    debugger.add_watchpoint("memory.context")

    # Register callbacks
    def on_step(snapshot):
        print(f"  Step {snapshot.step_number}: {snapshot.step_type} "
              f"({snapshot.tool_name or 'n/a'})")

    def on_breakpoint(bp, snapshot):
        print(f"  [BREAKPOINT HIT] {bp.id} at step {snapshot.step_number}")

    debugger.on_step(on_step)
    debugger.on_breakpoint(on_breakpoint)

    # Simulate agent execution
    print("\nSimulating agent execution...")
    print("-" * 40)

    agent_state = {
        "memory": {"context": [], "history": []},
        "tools_used": [],
        "turn_count": 0
    }

    # Step 1: LLM planning
    await debugger.record_step(
        state=agent_state,
        input_data={"user_query": "Search for Python tutorials"},
        output_data={"plan": ["search", "summarize"]},
        step_type="llm"
    )

    # Step 2: Tool call (search) - should hit breakpoint
    agent_state["tools_used"].append("search")
    await debugger.record_step(
        state=agent_state,
        input_data={"query": "Python tutorials"},
        output_data={"results": ["result1", "result2"]},
        step_type="tool",
        tool_name="search"
    )

    # Step 3: Update memory
    agent_state["memory"]["context"] = ["Python tutorials found"]
    await debugger.record_step(
        state=agent_state,
        input_data={"results": ["result1", "result2"]},
        output_data={"processed": True},
        step_type="internal"
    )

    # Step 4: Another LLM call
    await debugger.record_step(
        state=agent_state,
        input_data={"context": agent_state["memory"]["context"]},
        output_data={"summary": "Found Python tutorials"},
        step_type="llm"
    )

    # Step 5: Should hit conditional breakpoint
    agent_state["turn_count"] = 1
    await debugger.record_step(
        state=agent_state,
        input_data={"summary": "Found Python tutorials"},
        output_data={"response": "Here are Python tutorials..."},
        step_type="llm"
    )

    # Analysis
    print("\n" + "=" * 60)
    print("Execution Analysis")
    print("=" * 60)

    analysis = debugger.analyze_execution()
    print(f"\nTotal steps: {analysis['total_steps']}")
    print(f"Step types: {analysis['step_types']}")
    print(f"Tool usage: {analysis['tool_usage']}")
    print(f"Avg step duration: {analysis['avg_step_duration']:.4f}s")

    # State inspection
    print("\n" + "-" * 40)
    print("State Inspection")
    print("-" * 40)

    print(f"\nCurrent state keys: {list(debugger.inspect_state().keys())}")
    print(f"Memory context: {debugger.inspect_state('memory.context')}")

    # Diff between steps
    print("\n" + "-" * 40)
    print("State Diff (Step 2 -> Step 3)")
    print("-" * 40)

    diff = debugger.diff_snapshots(2, 3)
    for d in diff["differences"]:
        print(f"  {d['path']}: {d['change']}")

    # History
    print("\n" + "-" * 40)
    print("Execution History")
    print("-" * 40)

    for snapshot in debugger.get_history(5):
        print(f"  Step {snapshot.step_number}: {snapshot.step_type} - "
              f"{str(snapshot.output_data)[:50]}...")

    # Breakpoint stats
    print("\n" + "-" * 40)
    print("Breakpoint Statistics")
    print("-" * 40)

    for bp in debugger.breakpoints.values():
        print(f"  {bp.id}: hit {bp.hit_count} times")

    # Conversation debugger demo
    print("\n" + "=" * 60)
    print("Conversation Debugger Demo")
    print("=" * 60)

    conv_debugger = ConversationDebugger()

    conv_debugger.add_turn("user", "Can you search for Python tutorials?")
    conv_debugger.add_turn("assistant", "I'll search for Python tutorials for you.")
    conv_debugger.add_turn("tool", "Search results: 5 tutorials found")
    conv_debugger.add_turn("assistant", "I found 5 Python tutorials. Here are the top ones...")
    conv_debugger.add_turn("user", "Thanks! Can you explain the first one?")

    analysis = conv_debugger.analyze_conversation()
    print(f"\nConversation analysis:")
    print(f"  Total turns: {analysis['total_turns']}")
    print(f"  By role: {analysis['by_role']}")

    # Find specific turns
    user_turns = conv_debugger.find_turns(role="user")
    print(f"\nUser turns: {len(user_turns)}")
    for turn in user_turns:
        print(f"  - {turn['content'][:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
