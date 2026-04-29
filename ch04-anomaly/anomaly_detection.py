"""
Chapter 4: Audit Logging and Compliance
=======================================

Implements anomaly detection systems for monitoring AI agent behavior:
- Statistical anomaly detection
- Behavioral drift detection
- Cost anomaly detection
- Performance degradation detection
- Security anomaly detection

Based on observability best practices for production AI systems.
"""

import asyncio
import time
import math
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import deque, defaultdict


class AnomalyType(Enum):
    """Types of anomalies that can be detected."""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    COST = "cost"
    TOKEN_USAGE = "token_usage"
    BEHAVIORAL = "behavioral"
    SECURITY = "security"
    QUALITY = "quality"


class AlertSeverity(Enum):
    """Severity levels for anomaly alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    type: AnomalyType
    severity: AlertSeverity
    agent_id: str
    description: str
    expected_value: float
    actual_value: float
    deviation_score: float  # Standard deviations from mean
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def deviation_percentage(self) -> float:
        if self.expected_value == 0:
            return float('inf') if self.actual_value != 0 else 0
        return abs(self.actual_value - self.expected_value) / self.expected_value * 100


@dataclass
class MetricWindow:
    """Sliding window for metric collection."""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add(self, value: float, timestamp: Optional[float] = None):
        self.values.append(value)
        self.timestamps.append(timestamp or time.time())

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values) if self.values else 0.0

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * p / 100)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    def values_in_window(self, window_seconds: float) -> list[float]:
        cutoff = time.time() - window_seconds
        return [v for v, t in zip(self.values, self.timestamps) if t >= cutoff]


class AnomalyDetector(ABC):
    """Base class for anomaly detectors."""

    def __init__(self,
                 anomaly_type: AnomalyType,
                 warning_threshold: float = 2.0,
                 critical_threshold: float = 3.0):
        self.anomaly_type = anomaly_type
        self.warning_threshold = warning_threshold  # Standard deviations
        self.critical_threshold = critical_threshold
        self.enabled = True

    @abstractmethod
    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        """Detect if the value is anomalous."""
        pass

    def _calculate_severity(self, deviation_score: float) -> AlertSeverity:
        if abs(deviation_score) >= self.critical_threshold:
            return AlertSeverity.CRITICAL
        elif abs(deviation_score) >= self.warning_threshold:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO


class StatisticalDetector(AnomalyDetector):
    """
    Statistical anomaly detection using z-score and IQR methods.
    """

    def __init__(self,
                 anomaly_type: AnomalyType,
                 min_samples: int = 30,
                 **kwargs):
        super().__init__(anomaly_type, **kwargs)
        self.min_samples = min_samples
        self.windows: dict[str, MetricWindow] = defaultdict(MetricWindow)

    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        window = self.windows[agent_id]
        window.add(value)

        # Need minimum samples for reliable detection
        if len(window.values) < self.min_samples:
            return None

        # Z-score detection
        mean = window.mean
        std = window.std

        if std == 0:
            return None

        z_score = (value - mean) / std

        if abs(z_score) >= self.warning_threshold:
            return Anomaly(
                type=self.anomaly_type,
                severity=self._calculate_severity(z_score),
                agent_id=agent_id,
                description=f"Statistical anomaly detected: z-score={z_score:.2f}",
                expected_value=mean,
                actual_value=value,
                deviation_score=z_score,
                metadata={
                    "std": std,
                    "sample_size": len(window.values),
                    "method": "z_score",
                    **context
                }
            )
        return None


class IQRDetector(AnomalyDetector):
    """
    Interquartile range (IQR) based anomaly detection.
    More robust to outliers than z-score.
    """

    def __init__(self,
                 anomaly_type: AnomalyType,
                 min_samples: int = 30,
                 iqr_multiplier: float = 1.5,
                 **kwargs):
        super().__init__(anomaly_type, **kwargs)
        self.min_samples = min_samples
        self.iqr_multiplier = iqr_multiplier
        self.windows: dict[str, MetricWindow] = defaultdict(MetricWindow)

    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        window = self.windows[agent_id]
        window.add(value)

        if len(window.values) < self.min_samples:
            return None

        q1 = window.percentile(25)
        q3 = window.percentile(75)
        iqr = q3 - q1

        if iqr == 0:
            return None

        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr

        if value < lower_bound or value > upper_bound:
            median = window.median
            deviation = (value - median) / (iqr / 2) if iqr > 0 else 0

            return Anomaly(
                type=self.anomaly_type,
                severity=self._calculate_severity(deviation),
                agent_id=agent_id,
                description=f"IQR anomaly: value {value:.2f} outside bounds [{lower_bound:.2f}, {upper_bound:.2f}]",
                expected_value=median,
                actual_value=value,
                deviation_score=deviation,
                metadata={
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "method": "iqr",
                    **context
                }
            )
        return None


class ExponentialMovingAverageDetector(AnomalyDetector):
    """
    Exponential Moving Average (EMA) based detection.
    Good for detecting sudden changes.
    """

    def __init__(self,
                 anomaly_type: AnomalyType,
                 alpha: float = 0.1,
                 **kwargs):
        super().__init__(anomaly_type, **kwargs)
        self.alpha = alpha
        self.ema: dict[str, float] = {}
        self.ema_var: dict[str, float] = {}
        self.initialized: dict[str, bool] = {}

    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        if agent_id not in self.initialized:
            self.ema[agent_id] = value
            self.ema_var[agent_id] = 0.0
            self.initialized[agent_id] = True
            return None

        # Update EMA
        old_ema = self.ema[agent_id]
        self.ema[agent_id] = self.alpha * value + (1 - self.alpha) * old_ema

        # Update variance EMA
        squared_diff = (value - old_ema) ** 2
        self.ema_var[agent_id] = self.alpha * squared_diff + (1 - self.alpha) * self.ema_var[agent_id]

        # Calculate z-score based on EMA
        ema_std = math.sqrt(self.ema_var[agent_id])
        if ema_std == 0:
            return None

        z_score = (value - old_ema) / ema_std

        if abs(z_score) >= self.warning_threshold:
            return Anomaly(
                type=self.anomaly_type,
                severity=self._calculate_severity(z_score),
                agent_id=agent_id,
                description=f"EMA anomaly: sudden change detected (z={z_score:.2f})",
                expected_value=old_ema,
                actual_value=value,
                deviation_score=z_score,
                metadata={
                    "ema": self.ema[agent_id],
                    "ema_std": ema_std,
                    "alpha": self.alpha,
                    "method": "ema",
                    **context
                }
            )
        return None


class BehavioralDriftDetector(AnomalyDetector):
    """
    Detects behavioral drift in agent outputs using distribution comparison.
    """

    def __init__(self,
                 baseline_window: int = 1000,
                 current_window: int = 100,
                 **kwargs):
        super().__init__(AnomalyType.BEHAVIORAL, **kwargs)
        self.baseline_window = baseline_window
        self.current_window = current_window
        self.baselines: dict[str, list[float]] = defaultdict(list)
        self.current: dict[str, deque] = defaultdict(lambda: deque(maxlen=current_window))

    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        # Build baseline
        if len(self.baselines[agent_id]) < self.baseline_window:
            self.baselines[agent_id].append(value)
            return None

        # Track current window
        self.current[agent_id].append(value)

        if len(self.current[agent_id]) < self.current_window:
            return None

        # Compare distributions using KL divergence approximation
        baseline_mean = statistics.mean(self.baselines[agent_id])
        baseline_std = statistics.stdev(self.baselines[agent_id])
        current_mean = statistics.mean(self.current[agent_id])
        current_std = statistics.stdev(self.current[agent_id])

        if baseline_std == 0 or current_std == 0:
            return None

        # Simplified distribution comparison
        mean_drift = abs(current_mean - baseline_mean) / baseline_std
        variance_ratio = current_std / baseline_std

        drift_score = mean_drift + abs(math.log(variance_ratio)) if variance_ratio > 0 else mean_drift

        if drift_score >= self.warning_threshold:
            return Anomaly(
                type=self.anomaly_type,
                severity=self._calculate_severity(drift_score),
                agent_id=agent_id,
                description=f"Behavioral drift detected: distribution shift (score={drift_score:.2f})",
                expected_value=baseline_mean,
                actual_value=current_mean,
                deviation_score=drift_score,
                metadata={
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std,
                    "current_mean": current_mean,
                    "current_std": current_std,
                    "variance_ratio": variance_ratio,
                    "method": "distribution_drift",
                    **context
                }
            )
        return None


class CostAnomalyDetector(AnomalyDetector):
    """
    Specialized detector for cost anomalies.
    Tracks spending patterns and detects unusual costs.
    """

    def __init__(self,
                 daily_budget: float = 100.0,
                 hourly_budget: float = 10.0,
                 **kwargs):
        super().__init__(AnomalyType.COST, **kwargs)
        self.daily_budget = daily_budget
        self.hourly_budget = hourly_budget
        self.daily_costs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.hourly_costs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.statistical = StatisticalDetector(AnomalyType.COST, **kwargs)

    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        now = time.time()
        day_key = time.strftime("%Y-%m-%d", time.localtime(now))
        hour_key = time.strftime("%Y-%m-%d-%H", time.localtime(now))

        # Update accumulators
        self.daily_costs[agent_id][day_key] += value
        self.hourly_costs[agent_id][hour_key] += value

        daily_total = self.daily_costs[agent_id][day_key]
        hourly_total = self.hourly_costs[agent_id][hour_key]

        # Check budget thresholds
        if hourly_total > self.hourly_budget:
            severity = AlertSeverity.CRITICAL if hourly_total > self.hourly_budget * 2 else AlertSeverity.WARNING
            return Anomaly(
                type=self.anomaly_type,
                severity=severity,
                agent_id=agent_id,
                description=f"Hourly cost budget exceeded: ${hourly_total:.2f} > ${self.hourly_budget:.2f}",
                expected_value=self.hourly_budget,
                actual_value=hourly_total,
                deviation_score=hourly_total / self.hourly_budget,
                metadata={
                    "period": "hourly",
                    "hour_key": hour_key,
                    **context
                }
            )

        if daily_total > self.daily_budget:
            severity = AlertSeverity.CRITICAL if daily_total > self.daily_budget * 1.5 else AlertSeverity.WARNING
            return Anomaly(
                type=self.anomaly_type,
                severity=severity,
                agent_id=agent_id,
                description=f"Daily cost budget exceeded: ${daily_total:.2f} > ${self.daily_budget:.2f}",
                expected_value=self.daily_budget,
                actual_value=daily_total,
                deviation_score=daily_total / self.daily_budget,
                metadata={
                    "period": "daily",
                    "day_key": day_key,
                    **context
                }
            )

        # Also check for statistical anomalies in individual request costs
        return await self.statistical.detect(agent_id, value, context)


class SecurityAnomalyDetector(AnomalyDetector):
    """
    Detects security-related anomalies in agent behavior.
    """

    def __init__(self, **kwargs):
        super().__init__(AnomalyType.SECURITY, **kwargs)
        self.access_patterns: dict[str, list[dict]] = defaultdict(list)
        self.tool_usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.suspicious_patterns = [
            "rm -rf",
            "sudo",
            "chmod 777",
            "eval(",
            "exec(",
            "DROP TABLE",
            "DELETE FROM",
            "password",
            "secret",
            "api_key",
        ]

    async def detect(self, agent_id: str, value: float, context: dict) -> Optional[Anomaly]:
        # Track access patterns
        access_record = {
            "timestamp": time.time(),
            "resource": context.get("resource", ""),
            "action": context.get("action", ""),
            "tool": context.get("tool", "")
        }
        self.access_patterns[agent_id].append(access_record)

        # Keep only recent patterns
        cutoff = time.time() - 3600  # 1 hour
        self.access_patterns[agent_id] = [
            p for p in self.access_patterns[agent_id]
            if p["timestamp"] > cutoff
        ]

        # Check for suspicious content patterns
        content = context.get("content", "")
        for pattern in self.suspicious_patterns:
            if pattern.lower() in content.lower():
                return Anomaly(
                    type=self.anomaly_type,
                    severity=AlertSeverity.WARNING,
                    agent_id=agent_id,
                    description=f"Suspicious pattern detected: '{pattern}'",
                    expected_value=0,
                    actual_value=1,
                    deviation_score=3.0,  # Always warning level
                    metadata={
                        "pattern": pattern,
                        "content_snippet": content[:100],
                        **context
                    }
                )

        # Check for unusual access frequency
        tool = context.get("tool", "")
        if tool:
            self.tool_usage[agent_id][tool] += 1
            recent_count = sum(
                1 for p in self.access_patterns[agent_id]
                if p["tool"] == tool
            )

            # Rapid tool usage
            if recent_count > 100:
                return Anomaly(
                    type=self.anomaly_type,
                    severity=AlertSeverity.WARNING,
                    agent_id=agent_id,
                    description=f"Unusual tool usage frequency: {tool} used {recent_count} times in last hour",
                    expected_value=50,
                    actual_value=recent_count,
                    deviation_score=recent_count / 50,
                    metadata={
                        "tool": tool,
                        "count": recent_count,
                        **context
                    }
                )

        return None


class AnomalyMonitor:
    """
    Central anomaly monitoring system that coordinates multiple detectors.
    """

    def __init__(self):
        self.detectors: list[AnomalyDetector] = []
        self.anomalies: deque = deque(maxlen=10000)
        self.alert_callbacks: list[Callable] = []
        self._lock = asyncio.Lock()

    def add_detector(self, detector: AnomalyDetector):
        """Add an anomaly detector."""
        self.detectors.append(detector)

    def on_anomaly(self, callback: Callable):
        """Register callback for anomaly alerts."""
        self.alert_callbacks.append(callback)

    async def record_metric(self,
                            agent_id: str,
                            metric_type: AnomalyType,
                            value: float,
                            context: Optional[dict] = None) -> list[Anomaly]:
        """Record a metric and check for anomalies."""
        context = context or {}
        detected = []

        for detector in self.detectors:
            if detector.enabled and detector.anomaly_type == metric_type:
                anomaly = await detector.detect(agent_id, value, context)
                if anomaly:
                    detected.append(anomaly)
                    await self._handle_anomaly(anomaly)

        return detected

    async def record_request(self,
                              agent_id: str,
                              latency_ms: float,
                              tokens: int,
                              cost: float,
                              success: bool,
                              context: Optional[dict] = None) -> list[Anomaly]:
        """Convenience method to record common request metrics."""
        context = context or {}
        all_anomalies = []

        # Check latency
        anomalies = await self.record_metric(
            agent_id, AnomalyType.LATENCY, latency_ms, context
        )
        all_anomalies.extend(anomalies)

        # Check token usage
        anomalies = await self.record_metric(
            agent_id, AnomalyType.TOKEN_USAGE, tokens, context
        )
        all_anomalies.extend(anomalies)

        # Check cost
        anomalies = await self.record_metric(
            agent_id, AnomalyType.COST, cost, context
        )
        all_anomalies.extend(anomalies)

        # Check error rate (as binary 0/1)
        if not success:
            anomalies = await self.record_metric(
                agent_id, AnomalyType.ERROR_RATE, 1.0, context
            )
            all_anomalies.extend(anomalies)

        return all_anomalies

    async def _handle_anomaly(self, anomaly: Anomaly):
        """Handle a detected anomaly."""
        async with self._lock:
            self.anomalies.append(anomaly)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(anomaly)
                else:
                    callback(anomaly)
            except Exception:
                pass

    async def get_anomalies(self,
                            agent_id: Optional[str] = None,
                            anomaly_type: Optional[AnomalyType] = None,
                            severity: Optional[AlertSeverity] = None,
                            since: Optional[float] = None,
                            limit: int = 100) -> list[Anomaly]:
        """Query recorded anomalies."""
        results = []
        for anomaly in reversed(self.anomalies):
            if agent_id and anomaly.agent_id != agent_id:
                continue
            if anomaly_type and anomaly.type != anomaly_type:
                continue
            if severity and anomaly.severity != severity:
                continue
            if since and anomaly.timestamp < since:
                continue
            results.append(anomaly)
            if len(results) >= limit:
                break
        return results

    async def get_summary(self, window_seconds: int = 3600) -> dict:
        """Get summary of anomalies in time window."""
        cutoff = time.time() - window_seconds
        recent = [a for a in self.anomalies if a.timestamp >= cutoff]

        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_agent = defaultdict(int)

        for anomaly in recent:
            by_type[anomaly.type.value] += 1
            by_severity[anomaly.severity.value] += 1
            by_agent[anomaly.agent_id] += 1

        return {
            "total": len(recent),
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_agent": dict(by_agent),
            "window_seconds": window_seconds
        }


# =============================================================================
# Example Usage
# =============================================================================

import random

async def main():
    """Demonstration of anomaly detection."""
    print("=" * 60)
    print("Anomaly Detection Demonstration")
    print("=" * 60)

    # Create monitor with various detectors
    monitor = AnomalyMonitor()

    # Add detectors
    monitor.add_detector(StatisticalDetector(
        AnomalyType.LATENCY,
        min_samples=20,
        warning_threshold=2.0,
        critical_threshold=3.0
    ))

    monitor.add_detector(IQRDetector(
        AnomalyType.TOKEN_USAGE,
        min_samples=20
    ))

    monitor.add_detector(ExponentialMovingAverageDetector(
        AnomalyType.ERROR_RATE,
        alpha=0.2
    ))

    monitor.add_detector(CostAnomalyDetector(
        daily_budget=10.0,
        hourly_budget=2.0
    ))

    monitor.add_detector(SecurityAnomalyDetector())

    # Alert callback
    async def on_anomaly(anomaly: Anomaly):
        print(f"\n[ALERT] {anomaly.severity.value.upper()}: {anomaly.description}")
        print(f"        Agent: {anomaly.agent_id}, Type: {anomaly.type.value}")
        print(f"        Expected: {anomaly.expected_value:.2f}, Actual: {anomaly.actual_value:.2f}")

    monitor.on_anomaly(on_anomaly)

    print("\nSimulating agent metrics with occasional anomalies...")
    print("-" * 60)

    # Simulate normal operations
    for i in range(50):
        # Normal latency: ~100ms with some variance
        latency = random.gauss(100, 15)

        # Inject anomalies occasionally
        if i == 30:
            latency = 500  # Spike

        if i == 40:
            latency = 300  # Another spike

        await monitor.record_metric(
            "agent-1",
            AnomalyType.LATENCY,
            latency,
            {"request_id": f"req-{i}"}
        )

    # Simulate token usage with anomaly
    print("\n" + "-" * 60)
    print("Token usage anomaly test...")

    for i in range(50):
        tokens = random.gauss(1000, 100)
        if i == 45:
            tokens = 5000  # Anomalous token usage

        await monitor.record_metric(
            "agent-1",
            AnomalyType.TOKEN_USAGE,
            tokens,
            {"request_id": f"req-{i}"}
        )

    # Simulate cost exceeding budget
    print("\n" + "-" * 60)
    print("Cost budget test...")

    for i in range(30):
        cost = 0.05 + random.random() * 0.05  # $0.05-0.10 per request

        # Simulate expensive requests
        if i > 20:
            cost = 0.5  # More expensive

        await monitor.record_metric(
            "agent-2",
            AnomalyType.COST,
            cost,
            {"request_id": f"cost-{i}"}
        )

    # Simulate security anomaly
    print("\n" + "-" * 60)
    print("Security anomaly test...")

    await monitor.record_metric(
        "agent-3",
        AnomalyType.SECURITY,
        1.0,
        {
            "content": "Running command: rm -rf /tmp/test",
            "tool": "bash",
            "action": "execute"
        }
    )

    # Summary
    print("\n" + "=" * 60)
    print("Anomaly Summary")
    print("=" * 60)

    summary = await monitor.get_summary(window_seconds=3600)
    print(f"\nTotal anomalies: {summary['total']}")
    print(f"By type: {summary['by_type']}")
    print(f"By severity: {summary['by_severity']}")
    print(f"By agent: {summary['by_agent']}")

    # Query specific anomalies
    print("\n" + "-" * 60)
    print("Critical anomalies:")

    critical = await monitor.get_anomalies(severity=AlertSeverity.CRITICAL, limit=5)
    for anomaly in critical:
        print(f"  - [{anomaly.type.value}] {anomaly.description}")


if __name__ == "__main__":
    asyncio.run(main())
