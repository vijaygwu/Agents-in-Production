"""
Chapter 15: Agent Observability with OpenTelemetry
=================================================

Implements comprehensive observability for AI agents:
- Distributed tracing with spans
- Metrics collection and aggregation
- Structured logging
- Context propagation
- Agent-specific instrumentation

Based on OpenTelemetry standards and LangSmith/LangFuse patterns.
"""

import asyncio
import time
import uuid
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TypeVar, Generic
from collections import defaultdict
from contextlib import asynccontextmanager
from functools import wraps


# =============================================================================
# Core Types
# =============================================================================

class SpanKind(Enum):
    """Types of spans in a trace."""
    INTERNAL = "internal"
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    CHAIN = "chain"


class SpanStatus(Enum):
    """Status of a span."""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanContext:
    """Context for trace propagation."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: dict = field(default_factory=dict)

    def to_headers(self) -> dict:
        """Convert to HTTP headers for propagation."""
        return {
            "traceparent": f"00-{self.trace_id}-{self.span_id}-01",
            "tracestate": json.dumps(self.baggage)
        }

    @classmethod
    def from_headers(cls, headers: dict) -> 'SpanContext':
        """Create context from HTTP headers."""
        traceparent = headers.get("traceparent", "")
        parts = traceparent.split("-")

        if len(parts) >= 3:
            trace_id = parts[1]
            parent_span_id = parts[2]
        else:
            trace_id = uuid.uuid4().hex
            parent_span_id = None

        baggage = {}
        if "tracestate" in headers:
            try:
                baggage = json.loads(headers["tracestate"])
            except json.JSONDecodeError:
                pass

        return cls(
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:16],
            parent_span_id=parent_span_id,
            baggage=baggage
        )


@dataclass
class Span:
    """A span representing a unit of work in a trace."""
    span_id: str
    trace_id: str
    name: str
    kind: SpanKind
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    # Agent-specific fields
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    model: Optional[str] = None
    token_usage: Optional[dict] = None
    cost: Optional[float] = None

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    def set_attribute(self, key: str, value: Any):
        """Set a span attribute."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })

    def end(self, status: SpanStatus = SpanStatus.OK):
        """End the span."""
        self.end_time = time.time()
        self.status = status

    def to_dict(self) -> dict:
        """Convert span to dictionary for export."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "kind": self.kind.value,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "input": str(self.input_data)[:500] if self.input_data else None,
            "output": str(self.output_data)[:500] if self.output_data else None,
            "model": self.model,
            "token_usage": self.token_usage,
            "cost": self.cost
        }


@dataclass
class Metric:
    """A metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict = field(default_factory=dict)


@dataclass
class LogRecord:
    """A structured log record."""
    level: str
    message: str
    timestamp: float = field(default_factory=time.time)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    attributes: dict = field(default_factory=dict)


# =============================================================================
# Exporters
# =============================================================================

class SpanExporter(ABC):
    """Base class for span exporters."""

    @abstractmethod
    async def export(self, spans: list[Span]):
        """Export spans to a backend."""
        pass


class ConsoleSpanExporter(SpanExporter):
    """Export spans to console for debugging."""

    async def export(self, spans: list[Span]):
        for span in spans:
            print(f"[SPAN] {span.name} ({span.kind.value})")
            print(f"       trace_id={span.trace_id[:8]}... span_id={span.span_id[:8]}...")
            print(f"       duration={span.duration_ms:.2f}ms status={span.status.value}")
            if span.token_usage:
                print(f"       tokens={span.token_usage}")
            if span.cost:
                print(f"       cost=${span.cost:.4f}")


class InMemorySpanExporter(SpanExporter):
    """Export spans to in-memory storage."""

    def __init__(self):
        self.spans: list[Span] = []
        self.traces: dict[str, list[Span]] = defaultdict(list)

    async def export(self, spans: list[Span]):
        for span in spans:
            self.spans.append(span)
            self.traces[span.trace_id].append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        return self.traces.get(trace_id, [])

    def get_recent_spans(self, limit: int = 100) -> list[Span]:
        return self.spans[-limit:]


class MetricExporter(ABC):
    """Base class for metric exporters."""

    @abstractmethod
    async def export(self, metrics: list[Metric]):
        """Export metrics to a backend."""
        pass


class InMemoryMetricExporter(MetricExporter):
    """Export metrics to in-memory storage for aggregation."""

    def __init__(self):
        self.metrics: list[Metric] = []
        self.aggregations: dict[str, list[float]] = defaultdict(list)

    async def export(self, metrics: list[Metric]):
        for metric in metrics:
            self.metrics.append(metric)
            self.aggregations[metric.name].append(metric.value)

    def get_aggregation(self, name: str) -> dict:
        values = self.aggregations.get(name, [])
        if not values:
            return {}

        import statistics
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1]
        }


# =============================================================================
# Tracer
# =============================================================================

class Tracer:
    """
    Main tracing component for agent observability.
    """

    def __init__(self,
                 service_name: str = "agent-service",
                 span_exporters: Optional[list[SpanExporter]] = None):
        self.service_name = service_name
        self.span_exporters = span_exporters or [ConsoleSpanExporter()]
        self._current_context: dict[str, SpanContext] = {}  # task_id -> context
        self._pending_spans: list[Span] = []
        self._lock = asyncio.Lock()

    def _get_task_id(self) -> str:
        """Get current task identifier for context tracking."""
        try:
            task = asyncio.current_task()
            return str(id(task)) if task else "main"
        except RuntimeError:
            return "main"

    def get_current_context(self) -> Optional[SpanContext]:
        """Get the current span context."""
        return self._current_context.get(self._get_task_id())

    def set_current_context(self, context: SpanContext):
        """Set the current span context."""
        self._current_context[self._get_task_id()] = context

    @asynccontextmanager
    async def start_span(self,
                          name: str,
                          kind: SpanKind = SpanKind.INTERNAL,
                          attributes: Optional[dict] = None):
        """
        Context manager for creating and managing spans.

        Usage:
            async with tracer.start_span("my_operation") as span:
                span.set_attribute("key", "value")
                # ... do work ...
        """
        # Get or create trace context
        current_ctx = self.get_current_context()

        if current_ctx:
            trace_id = current_ctx.trace_id
            parent_span_id = current_ctx.span_id
        else:
            trace_id = uuid.uuid4().hex
            parent_span_id = None

        span_id = uuid.uuid4().hex[:16]

        # Create span
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            attributes=attributes or {}
        )

        # Set span context as current
        new_context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=current_ctx.baggage if current_ctx else {}
        )
        self.set_current_context(new_context)

        try:
            yield span
            span.end(SpanStatus.OK)
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.add_event("exception", {"message": str(e)})
            span.end(SpanStatus.ERROR)
            raise
        finally:
            # Restore parent context
            if current_ctx:
                self.set_current_context(current_ctx)
            elif self._get_task_id() in self._current_context:
                del self._current_context[self._get_task_id()]

            # Add span to pending and maybe flush
            async with self._lock:
                self._pending_spans.append(span)
                if len(self._pending_spans) >= 10:
                    await self._flush_spans()

    async def _flush_spans(self):
        """Flush pending spans to exporters."""
        if not self._pending_spans:
            return

        spans_to_export = self._pending_spans.copy()
        self._pending_spans.clear()

        for exporter in self.span_exporters:
            try:
                await exporter.export(spans_to_export)
            except Exception as e:
                print(f"Error exporting spans: {e}")

    async def flush(self):
        """Manually flush all pending spans."""
        async with self._lock:
            await self._flush_spans()


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Collects and aggregates metrics for agents.
    """

    def __init__(self,
                 metric_exporters: Optional[list[MetricExporter]] = None):
        self.metric_exporters = metric_exporters or []
        self._pending_metrics: list[Metric] = []
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def record_counter(self, name: str, value: float = 1.0,
                              unit: str = "1", attributes: Optional[dict] = None):
        """Record a counter metric (cumulative)."""
        async with self._lock:
            self._counters[name] += value
            metric = Metric(name=name, value=self._counters[name],
                          unit=unit, attributes=attributes or {})
            self._pending_metrics.append(metric)

    async def record_gauge(self, name: str, value: float,
                           unit: str = "1", attributes: Optional[dict] = None):
        """Record a gauge metric (point-in-time)."""
        async with self._lock:
            self._gauges[name] = value
            metric = Metric(name=name, value=value,
                          unit=unit, attributes=attributes or {})
            self._pending_metrics.append(metric)

    async def record_histogram(self, name: str, value: float,
                                unit: str = "1", attributes: Optional[dict] = None):
        """Record a histogram metric (distribution)."""
        async with self._lock:
            self._histograms[name].append(value)
            metric = Metric(name=name, value=value,
                          unit=unit, attributes=attributes or {})
            self._pending_metrics.append(metric)

    async def flush(self):
        """Flush pending metrics to exporters."""
        async with self._lock:
            if not self._pending_metrics:
                return

            metrics_to_export = self._pending_metrics.copy()
            self._pending_metrics.clear()

            for exporter in self.metric_exporters:
                try:
                    await exporter.export(metrics_to_export)
                except Exception as e:
                    print(f"Error exporting metrics: {e}")


# =============================================================================
# Logger
# =============================================================================

class StructuredLogger:
    """
    Structured logging with trace correlation.
    """

    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer
        self.records: list[LogRecord] = []

    def _get_trace_context(self) -> tuple[Optional[str], Optional[str]]:
        """Get current trace and span IDs."""
        if self.tracer:
            ctx = self.tracer.get_current_context()
            if ctx:
                return ctx.trace_id, ctx.span_id
        return None, None

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        trace_id, span_id = self._get_trace_context()

        record = LogRecord(
            level=level,
            message=message,
            trace_id=trace_id,
            span_id=span_id,
            attributes=kwargs
        )
        self.records.append(record)

        # Print to console
        ctx_str = f"[{trace_id[:8]}:{span_id[:8]}]" if trace_id and span_id else ""
        attr_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
        print(f"[{level}] {ctx_str} {message} {attr_str}")

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)


# =============================================================================
# Agent Instrumentation
# =============================================================================

class AgentInstrumentation:
    """
    High-level instrumentation for AI agents.
    Provides decorators and context managers for common operations.
    """

    def __init__(self,
                 service_name: str = "agent-service"):
        self.in_memory_exporter = InMemorySpanExporter()
        self.metric_exporter = InMemoryMetricExporter()

        self.tracer = Tracer(
            service_name=service_name,
            span_exporters=[ConsoleSpanExporter(), self.in_memory_exporter]
        )
        self.metrics = MetricsCollector(
            metric_exporters=[self.metric_exporter]
        )
        self.logger = StructuredLogger(self.tracer)

    def trace_llm_call(self, model: str = "unknown"):
        """Decorator for tracing LLM calls."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with self.tracer.start_span(
                    name=f"llm_call_{func.__name__}",
                    kind=SpanKind.LLM,
                    attributes={"model": model}
                ) as span:
                    span.model = model
                    span.input_data = kwargs.get("prompt") or (args[0] if args else None)

                    start = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start

                    span.output_data = result

                    # Extract token usage if available
                    if isinstance(result, dict):
                        if "usage" in result:
                            span.token_usage = result["usage"]
                            total_tokens = result["usage"].get("total_tokens", 0)
                            span.cost = total_tokens * 0.00001  # Example cost

                    # Record metrics
                    await self.metrics.record_histogram(
                        "llm.latency", duration * 1000, unit="ms",
                        attributes={"model": model}
                    )
                    await self.metrics.record_counter(
                        "llm.calls", 1,
                        attributes={"model": model}
                    )

                    return result
            return wrapper
        return decorator

    def trace_tool_call(self, tool_name: str):
        """Decorator for tracing tool calls."""
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                async with self.tracer.start_span(
                    name=f"tool_{tool_name}",
                    kind=SpanKind.TOOL,
                    attributes={"tool.name": tool_name}
                ) as span:
                    span.input_data = {"args": args, "kwargs": kwargs}

                    start = time.time()
                    result = await func(*args, **kwargs)
                    duration = time.time() - start

                    span.output_data = result

                    await self.metrics.record_histogram(
                        "tool.latency", duration * 1000, unit="ms",
                        attributes={"tool": tool_name}
                    )
                    await self.metrics.record_counter(
                        "tool.calls", 1,
                        attributes={"tool": tool_name}
                    )

                    return result
            return wrapper
        return decorator

    @asynccontextmanager
    async def trace_agent_turn(self, agent_id: str, turn_id: Optional[str] = None):
        """Context manager for tracing a complete agent turn."""
        turn_id = turn_id or uuid.uuid4().hex[:12]

        async with self.tracer.start_span(
            name=f"agent_turn_{agent_id}",
            kind=SpanKind.AGENT,
            attributes={
                "agent.id": agent_id,
                "turn.id": turn_id
            }
        ) as span:
            self.logger.info(f"Starting agent turn", agent_id=agent_id, turn_id=turn_id)

            yield span

            self.logger.info(
                f"Completed agent turn",
                agent_id=agent_id,
                turn_id=turn_id,
                duration_ms=span.duration_ms
            )

            await self.metrics.record_histogram(
                "agent.turn.latency", span.duration_ms, unit="ms",
                attributes={"agent_id": agent_id}
            )

    @asynccontextmanager
    async def trace_retrieval(self, query: str, source: str = "default"):
        """Context manager for tracing retrieval operations."""
        async with self.tracer.start_span(
            name=f"retrieval_{source}",
            kind=SpanKind.RETRIEVAL,
            attributes={
                "retrieval.source": source,
                "retrieval.query": query[:100]
            }
        ) as span:
            span.input_data = {"query": query}
            yield span

            await self.metrics.record_counter(
                "retrieval.calls", 1,
                attributes={"source": source}
            )

    async def get_trace_summary(self, trace_id: str) -> dict:
        """Get summary of a trace."""
        spans = self.in_memory_exporter.get_trace(trace_id)

        if not spans:
            return {"error": "Trace not found"}

        total_duration = sum(s.duration_ms for s in spans)
        total_cost = sum(s.cost or 0 for s in spans)
        total_tokens = sum(
            (s.token_usage or {}).get("total_tokens", 0)
            for s in spans
        )

        return {
            "trace_id": trace_id,
            "span_count": len(spans),
            "total_duration_ms": total_duration,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "spans": [s.to_dict() for s in spans]
        }

    async def get_metrics_summary(self) -> dict:
        """Get summary of collected metrics."""
        return {
            name: self.metric_exporter.get_aggregation(name)
            for name in self.metric_exporter.aggregations.keys()
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of agent observability."""
    print("=" * 60)
    print("Agent Observability Demonstration")
    print("=" * 60)

    # Create instrumentation
    instrumentation = AgentInstrumentation(service_name="demo-agent")

    # Define instrumented functions
    @instrumentation.trace_llm_call(model="gpt-4")
    async def call_llm(prompt: str) -> dict:
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "response": f"Response to: {prompt}",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            }
        }

    @instrumentation.trace_tool_call("search")
    async def search_tool(query: str) -> list:
        await asyncio.sleep(0.05)
        return [f"Result 1 for {query}", f"Result 2 for {query}"]

    @instrumentation.trace_tool_call("calculator")
    async def calculator_tool(expression: str) -> float:
        await asyncio.sleep(0.01)
        return eval(expression)  # Demo only

    # Run agent workflow
    print("\n" + "-" * 40)
    print("Running instrumented agent workflow...")
    print("-" * 40 + "\n")

    async with instrumentation.trace_agent_turn("demo-agent-1", "turn-001") as turn_span:
        turn_span.input_data = {"user_query": "What is 2+2 and search for Python"}

        # Retrieval
        async with instrumentation.trace_retrieval("Python programming", "vector_db") as ret_span:
            search_results = await search_tool("Python")
            ret_span.output_data = search_results

        # LLM call
        llm_response = await call_llm(f"Answer based on: {search_results}")

        # Tool call
        calc_result = await calculator_tool("2+2")

        turn_span.output_data = {
            "llm_response": llm_response,
            "calculation": calc_result
        }

    # Flush all telemetry
    await instrumentation.tracer.flush()
    await instrumentation.metrics.flush()

    # Get summaries
    print("\n" + "=" * 60)
    print("Telemetry Summary")
    print("=" * 60)

    # Recent spans
    recent_spans = instrumentation.in_memory_exporter.get_recent_spans(limit=10)
    print(f"\nRecent spans: {len(recent_spans)}")
    for span in recent_spans:
        print(f"  - {span.name}: {span.duration_ms:.2f}ms ({span.status.value})")

    # Metrics summary
    print("\nMetrics:")
    metrics_summary = await instrumentation.get_metrics_summary()
    for metric_name, stats in metrics_summary.items():
        if stats:
            print(f"  {metric_name}:")
            print(f"    count={stats.get('count', 0)}, mean={stats.get('mean', 0):.2f}")

    # Get specific trace
    if recent_spans:
        trace_id = recent_spans[0].trace_id
        print(f"\nTrace summary for {trace_id[:8]}...:")
        trace_summary = await instrumentation.get_trace_summary(trace_id)
        print(f"  Span count: {trace_summary['span_count']}")
        print(f"  Total duration: {trace_summary['total_duration_ms']:.2f}ms")
        print(f"  Total cost: ${trace_summary['total_cost']:.4f}")
        print(f"  Total tokens: {trace_summary['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
