"""
Chapter 8: Testing Agent Systems
================================

Implements cost tracking and attribution for AI agent systems:
- Token-level cost tracking
- Per-agent cost attribution
- Budget management and alerts
- Cost forecasting
- Chargeback reporting

Essential for enterprise AI agent deployments.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import json


class CostCategory(Enum):
    """Categories of costs in agent systems."""
    LLM_INPUT = "llm_input"
    LLM_OUTPUT = "llm_output"
    EMBEDDING = "embedding"
    TOOL_EXECUTION = "tool_execution"
    RETRIEVAL = "retrieval"
    COMPUTE = "compute"
    STORAGE = "storage"
    NETWORK = "network"


class AlertLevel(Enum):
    """Alert levels for budget monitoring."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ModelPricing:
    """Pricing configuration for a model."""
    model_id: str
    input_cost_per_1k: float   # Cost per 1000 input tokens
    output_cost_per_1k: float  # Cost per 1000 output tokens
    embedding_cost_per_1k: float = 0.0
    effective_date: str = ""

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a request."""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return input_cost + output_cost


@dataclass
class CostEntry:
    """A single cost entry."""
    id: str
    timestamp: float
    category: CostCategory
    amount: float
    currency: str = "USD"
    agent_id: str = ""
    session_id: str = ""
    user_id: str = ""
    project_id: str = ""
    model_id: str = ""
    metadata: dict = field(default_factory=dict)

    # Detailed breakdown
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0


@dataclass
class Budget:
    """Budget configuration."""
    id: str
    name: str
    amount: float
    period: str  # daily, weekly, monthly, total
    currency: str = "USD"
    scope_type: str = "global"  # global, project, agent, user
    scope_id: str = ""
    warning_threshold: float = 0.8  # 80% of budget
    critical_threshold: float = 0.95  # 95% of budget
    enabled: bool = True


@dataclass
class BudgetAlert:
    """Budget alert notification."""
    budget_id: str
    level: AlertLevel
    message: str
    current_spend: float
    budget_amount: float
    percentage: float
    timestamp: float = field(default_factory=time.time)


class PricingRegistry:
    """
    Registry of model pricing configurations.
    """

    def __init__(self):
        self.models: dict[str, ModelPricing] = {}
        self._load_default_pricing()

    def _load_default_pricing(self):
        """Load default pricing for common models."""
        default_models = [
            ModelPricing("gpt-4", 0.03, 0.06),
            ModelPricing("gpt-4-turbo", 0.01, 0.03),
            ModelPricing("gpt-4o", 0.005, 0.015),
            ModelPricing("gpt-3.5-turbo", 0.0005, 0.0015),
            ModelPricing("claude-3-opus", 0.015, 0.075),
            ModelPricing("claude-3-sonnet", 0.003, 0.015),
            ModelPricing("claude-3-haiku", 0.00025, 0.00125),
            ModelPricing("claude-3.5-sonnet", 0.003, 0.015),
            ModelPricing("gemini-1.5-pro", 0.0035, 0.0105),
            ModelPricing("gemini-1.5-flash", 0.00035, 0.00105),
            ModelPricing("text-embedding-3-small", 0.00002, 0.0, 0.00002),
            ModelPricing("text-embedding-3-large", 0.00013, 0.0, 0.00013),
        ]

        for model in default_models:
            self.models[model.model_id] = model

    def get_pricing(self, model_id: str) -> Optional[ModelPricing]:
        """Get pricing for a model."""
        # Try exact match
        if model_id in self.models:
            return self.models[model_id]

        # Try prefix match
        for key, pricing in self.models.items():
            if model_id.startswith(key):
                return pricing

        return None

    def add_pricing(self, pricing: ModelPricing):
        """Add or update model pricing."""
        self.models[pricing.model_id] = pricing


class CostTracker:
    """
    Tracks and manages costs for AI agent operations.
    """

    def __init__(self):
        self.pricing_registry = PricingRegistry()
        self.entries: list[CostEntry] = []
        self.budgets: dict[str, Budget] = {}
        self.alerts: list[BudgetAlert] = []
        self._alert_callbacks: list[Callable] = []
        self._lock = asyncio.Lock()

        # Aggregation caches
        self._agent_totals: dict[str, float] = defaultdict(float)
        self._project_totals: dict[str, float] = defaultdict(float)
        self._user_totals: dict[str, float] = defaultdict(float)
        self._daily_totals: dict[str, float] = defaultdict(float)

    async def record_llm_cost(self,
                               model_id: str,
                               input_tokens: int,
                               output_tokens: int,
                               agent_id: str = "",
                               session_id: str = "",
                               user_id: str = "",
                               project_id: str = "",
                               metadata: Optional[dict] = None) -> CostEntry:
        """Record cost for an LLM call."""
        pricing = self.pricing_registry.get_pricing(model_id)

        if pricing:
            cost = pricing.calculate_cost(input_tokens, output_tokens)
        else:
            # Default fallback pricing
            cost = (input_tokens + output_tokens) * 0.00001

        entry = CostEntry(
            id=f"cost_{int(time.time()*1000)}_{len(self.entries)}",
            timestamp=time.time(),
            category=CostCategory.LLM_INPUT,
            amount=cost,
            agent_id=agent_id,
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata=metadata or {}
        )

        await self._add_entry(entry)
        return entry

    async def record_embedding_cost(self,
                                     model_id: str,
                                     tokens: int,
                                     **kwargs) -> CostEntry:
        """Record cost for embedding operation."""
        pricing = self.pricing_registry.get_pricing(model_id)

        if pricing:
            cost = (tokens / 1000) * pricing.embedding_cost_per_1k
        else:
            cost = tokens * 0.000001

        entry = CostEntry(
            id=f"cost_{int(time.time()*1000)}_{len(self.entries)}",
            timestamp=time.time(),
            category=CostCategory.EMBEDDING,
            amount=cost,
            model_id=model_id,
            embedding_tokens=tokens,
            **kwargs
        )

        await self._add_entry(entry)
        return entry

    async def record_tool_cost(self,
                                tool_name: str,
                                cost: float,
                                **kwargs) -> CostEntry:
        """Record cost for tool execution."""
        entry = CostEntry(
            id=f"cost_{int(time.time()*1000)}_{len(self.entries)}",
            timestamp=time.time(),
            category=CostCategory.TOOL_EXECUTION,
            amount=cost,
            metadata={"tool_name": tool_name, **kwargs.get("metadata", {})},
            **{k: v for k, v in kwargs.items() if k != "metadata"}
        )

        await self._add_entry(entry)
        return entry

    async def _add_entry(self, entry: CostEntry):
        """Add a cost entry and update aggregations."""
        async with self._lock:
            self.entries.append(entry)

            # Update aggregations
            if entry.agent_id:
                self._agent_totals[entry.agent_id] += entry.amount
            if entry.project_id:
                self._project_totals[entry.project_id] += entry.amount
            if entry.user_id:
                self._user_totals[entry.user_id] += entry.amount

            day_key = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
            self._daily_totals[day_key] += entry.amount

        # Check budgets
        await self._check_budgets(entry)

    async def _check_budgets(self, entry: CostEntry):
        """Check if any budgets are exceeded."""
        for budget in self.budgets.values():
            if not budget.enabled:
                continue

            spend = await self._get_budget_spend(budget)
            percentage = spend / budget.amount if budget.amount > 0 else 0

            if percentage >= budget.critical_threshold:
                await self._create_alert(budget, AlertLevel.CRITICAL, spend, percentage)
            elif percentage >= budget.warning_threshold:
                await self._create_alert(budget, AlertLevel.WARNING, spend, percentage)

    async def _get_budget_spend(self, budget: Budget) -> float:
        """Calculate current spend for a budget."""
        now = time.time()

        # Determine time window
        if budget.period == "daily":
            start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        elif budget.period == "weekly":
            start = (datetime.now() - timedelta(days=datetime.now().weekday())).replace(
                hour=0, minute=0, second=0
            ).timestamp()
        elif budget.period == "monthly":
            start = datetime.now().replace(day=1, hour=0, minute=0, second=0).timestamp()
        else:  # total
            start = 0

        # Filter entries
        relevant = [e for e in self.entries if e.timestamp >= start]

        # Apply scope
        if budget.scope_type == "agent":
            relevant = [e for e in relevant if e.agent_id == budget.scope_id]
        elif budget.scope_type == "project":
            relevant = [e for e in relevant if e.project_id == budget.scope_id]
        elif budget.scope_type == "user":
            relevant = [e for e in relevant if e.user_id == budget.scope_id]

        return sum(e.amount for e in relevant)

    async def _create_alert(self, budget: Budget, level: AlertLevel,
                            spend: float, percentage: float):
        """Create a budget alert."""
        alert = BudgetAlert(
            budget_id=budget.id,
            level=level,
            message=f"Budget '{budget.name}' at {percentage:.1%} ({spend:.2f}/{budget.amount:.2f})",
            current_spend=spend,
            budget_amount=budget.amount,
            percentage=percentage
        )

        self.alerts.append(alert)

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception:
                pass

    def on_budget_alert(self, callback: Callable):
        """Register callback for budget alerts."""
        self._alert_callbacks.append(callback)

    # ==========================================================================
    # Budget Management
    # ==========================================================================

    def add_budget(self, budget: Budget) -> str:
        """Add a budget."""
        self.budgets[budget.id] = budget
        return budget.id

    def update_budget(self, budget_id: str, **updates) -> bool:
        """Update a budget."""
        if budget_id not in self.budgets:
            return False

        budget = self.budgets[budget_id]
        for key, value in updates.items():
            if hasattr(budget, key):
                setattr(budget, key, value)
        return True

    def remove_budget(self, budget_id: str) -> bool:
        """Remove a budget."""
        if budget_id in self.budgets:
            del self.budgets[budget_id]
            return True
        return False

    # ==========================================================================
    # Reporting
    # ==========================================================================

    async def get_summary(self,
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None,
                           group_by: Optional[str] = None) -> dict:
        """Get cost summary."""
        entries = self.entries

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        total = sum(e.amount for e in entries)
        total_input_tokens = sum(e.input_tokens for e in entries)
        total_output_tokens = sum(e.output_tokens for e in entries)

        summary = {
            "total_cost": total,
            "entry_count": len(entries),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "by_category": {},
            "by_model": {}
        }

        # Group by category
        for entry in entries:
            cat = entry.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + entry.amount

            if entry.model_id:
                summary["by_model"][entry.model_id] = (
                    summary["by_model"].get(entry.model_id, 0) + entry.amount
                )

        # Optional additional grouping
        if group_by == "agent":
            summary["by_agent"] = dict(self._agent_totals)
        elif group_by == "project":
            summary["by_project"] = dict(self._project_totals)
        elif group_by == "user":
            summary["by_user"] = dict(self._user_totals)
        elif group_by == "day":
            summary["by_day"] = dict(self._daily_totals)

        return summary

    async def get_agent_costs(self, agent_id: str,
                               start_time: Optional[float] = None) -> dict:
        """Get detailed costs for an agent."""
        entries = [e for e in self.entries if e.agent_id == agent_id]

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]

        return {
            "agent_id": agent_id,
            "total_cost": sum(e.amount for e in entries),
            "total_requests": len(entries),
            "total_input_tokens": sum(e.input_tokens for e in entries),
            "total_output_tokens": sum(e.output_tokens for e in entries),
            "by_model": self._group_by_model(entries),
            "by_category": self._group_by_category(entries),
            "by_day": self._group_by_day(entries)
        }

    def _group_by_model(self, entries: list[CostEntry]) -> dict:
        groups = defaultdict(lambda: {"cost": 0, "requests": 0})
        for e in entries:
            if e.model_id:
                groups[e.model_id]["cost"] += e.amount
                groups[e.model_id]["requests"] += 1
        return dict(groups)

    def _group_by_category(self, entries: list[CostEntry]) -> dict:
        groups = defaultdict(float)
        for e in entries:
            groups[e.category.value] += e.amount
        return dict(groups)

    def _group_by_day(self, entries: list[CostEntry]) -> dict:
        groups = defaultdict(float)
        for e in entries:
            day = datetime.fromtimestamp(e.timestamp).strftime("%Y-%m-%d")
            groups[day] += e.amount
        return dict(groups)

    async def generate_chargeback_report(self,
                                          period_start: float,
                                          period_end: float) -> dict:
        """Generate chargeback report for billing."""
        entries = [
            e for e in self.entries
            if period_start <= e.timestamp <= period_end
        ]

        # Group by project and user
        by_project = defaultdict(lambda: {
            "total": 0,
            "by_user": defaultdict(float),
            "by_agent": defaultdict(float)
        })

        for entry in entries:
            project = entry.project_id or "unassigned"
            by_project[project]["total"] += entry.amount
            if entry.user_id:
                by_project[project]["by_user"][entry.user_id] += entry.amount
            if entry.agent_id:
                by_project[project]["by_agent"][entry.agent_id] += entry.amount

        return {
            "period_start": datetime.fromtimestamp(period_start).isoformat(),
            "period_end": datetime.fromtimestamp(period_end).isoformat(),
            "total_cost": sum(e.amount for e in entries),
            "by_project": {
                k: {
                    "total": v["total"],
                    "by_user": dict(v["by_user"]),
                    "by_agent": dict(v["by_agent"])
                }
                for k, v in by_project.items()
            }
        }

    # ==========================================================================
    # Forecasting
    # ==========================================================================

    async def forecast_costs(self, days_ahead: int = 30) -> dict:
        """Simple cost forecasting based on recent trends."""
        # Get last 30 days of data
        now = time.time()
        thirty_days_ago = now - (30 * 24 * 60 * 60)

        daily_costs = []
        for day_key, amount in self._daily_totals.items():
            day_ts = datetime.strptime(day_key, "%Y-%m-%d").timestamp()
            if day_ts >= thirty_days_ago:
                daily_costs.append(amount)

        if not daily_costs:
            return {"error": "Insufficient data for forecasting"}

        avg_daily = sum(daily_costs) / len(daily_costs)
        forecast = avg_daily * days_ahead

        # Simple trend calculation
        if len(daily_costs) >= 7:
            recent_avg = sum(daily_costs[-7:]) / 7
            older_avg = sum(daily_costs[:-7]) / max(len(daily_costs) - 7, 1)
            trend = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
        else:
            trend = 0

        return {
            "forecast_days": days_ahead,
            "avg_daily_cost": avg_daily,
            "forecasted_total": forecast,
            "trend_percentage": trend * 100,
            "confidence": "low" if len(daily_costs) < 14 else "medium"
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of cost attribution."""
    print("=" * 60)
    print("Cost Attribution Demonstration")
    print("=" * 60)

    # Create tracker
    tracker = CostTracker()

    # Set up budget
    budget = Budget(
        id="project-alpha-monthly",
        name="Project Alpha Monthly Budget",
        amount=100.0,
        period="monthly",
        scope_type="project",
        scope_id="alpha",
        warning_threshold=0.5,
        critical_threshold=0.8
    )
    tracker.add_budget(budget)

    # Alert callback
    async def on_alert(alert: BudgetAlert):
        print(f"\n[ALERT] {alert.level.value.upper()}: {alert.message}")

    tracker.on_budget_alert(on_alert)

    print("\nSimulating agent operations...")
    print("-" * 40)

    # Simulate various costs
    agents = ["research-agent", "writing-agent", "analysis-agent"]
    models = ["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"]

    for i in range(20):
        agent = agents[i % len(agents)]
        model = models[i % len(models)]

        # Record LLM cost
        entry = await tracker.record_llm_cost(
            model_id=model,
            input_tokens=500 + i * 100,
            output_tokens=200 + i * 50,
            agent_id=agent,
            project_id="alpha",
            user_id="user-1",
            session_id=f"session-{i // 5}"
        )

        print(f"  {agent} ({model}): ${entry.amount:.4f}")

        # Occasionally record tool costs
        if i % 3 == 0:
            await tracker.record_tool_cost(
                tool_name="web_search",
                cost=0.01,
                agent_id=agent,
                project_id="alpha"
            )

        # Record embedding costs
        if i % 5 == 0:
            await tracker.record_embedding_cost(
                model_id="text-embedding-3-small",
                tokens=1000,
                agent_id=agent,
                project_id="alpha"
            )

    # Get summaries
    print("\n" + "=" * 60)
    print("Cost Summary")
    print("=" * 60)

    summary = await tracker.get_summary(group_by="agent")
    print(f"\nTotal cost: ${summary['total_cost']:.4f}")
    print(f"Total entries: {summary['entry_count']}")
    print(f"Total tokens: {summary['total_input_tokens'] + summary['total_output_tokens']}")

    print("\nBy Category:")
    for cat, cost in summary['by_category'].items():
        print(f"  {cat}: ${cost:.4f}")

    print("\nBy Model:")
    for model, cost in summary['by_model'].items():
        print(f"  {model}: ${cost:.4f}")

    print("\nBy Agent:")
    for agent, cost in summary.get('by_agent', {}).items():
        print(f"  {agent}: ${cost:.4f}")

    # Agent-specific report
    print("\n" + "-" * 40)
    print("Research Agent Detailed Costs")
    print("-" * 40)

    agent_costs = await tracker.get_agent_costs("research-agent")
    print(f"\nTotal: ${agent_costs['total_cost']:.4f}")
    print(f"Requests: {agent_costs['total_requests']}")
    print(f"Input tokens: {agent_costs['total_input_tokens']}")
    print(f"Output tokens: {agent_costs['total_output_tokens']}")

    # Chargeback report
    print("\n" + "-" * 40)
    print("Chargeback Report (simulated)")
    print("-" * 40)

    now = time.time()
    report = await tracker.generate_chargeback_report(
        period_start=now - 3600,
        period_end=now
    )

    print(f"\nPeriod: {report['period_start']} to {report['period_end']}")
    print(f"Total: ${report['total_cost']:.4f}")

    for project, data in report['by_project'].items():
        print(f"\n  Project '{project}': ${data['total']:.4f}")
        print(f"    By agent: {data['by_agent']}")

    # Forecast
    print("\n" + "-" * 40)
    print("Cost Forecast")
    print("-" * 40)

    forecast = await tracker.forecast_costs(days_ahead=30)
    if "error" not in forecast:
        print(f"\nAverage daily cost: ${forecast['avg_daily_cost']:.4f}")
        print(f"30-day forecast: ${forecast['forecasted_total']:.4f}")
        print(f"Trend: {forecast['trend_percentage']:+.1f}%")
        print(f"Confidence: {forecast['confidence']}")

    # Budget status
    print("\n" + "-" * 40)
    print("Budget Status")
    print("-" * 40)

    for budget in tracker.budgets.values():
        spend = await tracker._get_budget_spend(budget)
        pct = (spend / budget.amount) * 100 if budget.amount > 0 else 0
        print(f"\n{budget.name}:")
        print(f"  Spent: ${spend:.2f} / ${budget.amount:.2f} ({pct:.1f}%)")
        print(f"  Status: {'WARNING' if pct >= budget.warning_threshold * 100 else 'OK'}")

    # Recent alerts
    if tracker.alerts:
        print("\n" + "-" * 40)
        print("Recent Alerts")
        print("-" * 40)
        for alert in tracker.alerts[-5:]:
            print(f"  [{alert.level.value}] {alert.message}")


if __name__ == "__main__":
    asyncio.run(main())
