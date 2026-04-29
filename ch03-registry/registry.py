"""
Chapter 3: Rate Limiting and Cost Control
=========================================

Implements a service registry for AI agents providing:
- Agent registration and discovery
- Capability-based matching
- Health monitoring
- Version management
- Dependency tracking

Based on microservices service discovery patterns adapted for AI agents.
"""

import asyncio
import time
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from collections import defaultdict


class AgentStatus(Enum):
    """Agent health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPING = "stopping"
    UNKNOWN = "unknown"


class CapabilityType(Enum):
    """Types of agent capabilities."""
    TOOL = "tool"           # Can use specific tools
    SKILL = "skill"         # Has specific skills (analysis, writing, etc.)
    MODEL = "model"         # Uses specific model type
    DOMAIN = "domain"       # Domain expertise
    PROTOCOL = "protocol"   # Supports specific protocols


@dataclass
class Capability:
    """A capability that an agent possesses."""
    type: CapabilityType
    name: str
    version: str = "1.0.0"
    parameters: dict = field(default_factory=dict)
    performance_score: float = 0.0  # 0.0 to 1.0

    def matches(self, requirement: 'CapabilityRequirement') -> bool:
        """Check if this capability matches a requirement."""
        if self.type != requirement.type or self.name != requirement.name:
            return False

        if requirement.min_version:
            if not self._version_gte(self.version, requirement.min_version):
                return False

        if requirement.min_performance and self.performance_score < requirement.min_performance:
            return False

        return True

    def _version_gte(self, v1: str, v2: str) -> bool:
        """Check if v1 >= v2."""
        def parse(v):
            return tuple(int(x) for x in v.split('.'))
        return parse(v1) >= parse(v2)


@dataclass
class CapabilityRequirement:
    """A requirement for agent capabilities."""
    type: CapabilityType
    name: str
    min_version: Optional[str] = None
    min_performance: Optional[float] = None
    required: bool = True


@dataclass
class AgentEndpoint:
    """Connection endpoint for an agent."""
    protocol: str  # http, grpc, mcp, a2a
    address: str
    port: int
    path: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        base = f"{self.protocol}://{self.address}:{self.port}"
        if self.path:
            return f"{base}/{self.path.lstrip('/')}"
        return base


@dataclass
class HealthCheck:
    """Health check configuration."""
    interval_seconds: int = 30
    timeout_seconds: int = 5
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    endpoint: Optional[str] = None


@dataclass
class AgentRegistration:
    """Complete agent registration information."""
    agent_id: str
    name: str
    description: str
    version: str
    capabilities: list[Capability]
    endpoints: list[AgentEndpoint]
    health_check: HealthCheck = field(default_factory=HealthCheck)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # Other agent IDs
    max_concurrent: int = 10
    registered_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    status: AgentStatus = AgentStatus.STARTING
    instance_id: str = ""

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = hashlib.sha256(
                f"{self.agent_id}:{self.registered_at}".encode()
            ).hexdigest()[:12]


@dataclass
class HealthState:
    """Health state tracking for an agent."""
    status: AgentStatus = AgentStatus.UNKNOWN
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_check: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    latency_ms: float = 0.0
    error_message: str = ""


class AgentRegistry:
    """
    Central registry for AI agents.
    Provides registration, discovery, and health monitoring.
    """

    def __init__(self):
        self.agents: dict[str, AgentRegistration] = {}
        self.health_states: dict[str, HealthState] = {}
        self.capability_index: dict[str, set[str]] = defaultdict(set)  # capability_key -> agent_ids
        self.tag_index: dict[str, set[str]] = defaultdict(set)  # tag -> agent_ids
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._health_callbacks: list[Callable] = []

    async def register(self, registration: AgentRegistration) -> dict:
        """Register an agent with the registry."""
        async with self._lock:
            # Check for duplicate
            if registration.agent_id in self.agents:
                existing = self.agents[registration.agent_id]
                if existing.instance_id != registration.instance_id:
                    # Different instance - update
                    await self._deregister_internal(registration.agent_id)

            # Store registration
            self.agents[registration.agent_id] = registration
            self.health_states[registration.agent_id] = HealthState()

            # Index capabilities
            for capability in registration.capabilities:
                key = f"{capability.type.value}:{capability.name}"
                self.capability_index[key].add(registration.agent_id)

            # Index tags
            for tag in registration.tags:
                self.tag_index[tag].add(registration.agent_id)

            return {
                "status": "registered",
                "agent_id": registration.agent_id,
                "instance_id": registration.instance_id
            }

    async def deregister(self, agent_id: str) -> dict:
        """Deregister an agent from the registry."""
        async with self._lock:
            if agent_id not in self.agents:
                return {"status": "not_found", "agent_id": agent_id}

            await self._deregister_internal(agent_id)
            return {"status": "deregistered", "agent_id": agent_id}

    async def _deregister_internal(self, agent_id: str):
        """Internal deregistration without lock."""
        if agent_id in self.agents:
            registration = self.agents[agent_id]

            # Remove from capability index
            for capability in registration.capabilities:
                key = f"{capability.type.value}:{capability.name}"
                self.capability_index[key].discard(agent_id)

            # Remove from tag index
            for tag in registration.tags:
                self.tag_index[tag].discard(agent_id)

            del self.agents[agent_id]
            if agent_id in self.health_states:
                del self.health_states[agent_id]

    async def heartbeat(self, agent_id: str, status_update: Optional[dict] = None) -> dict:
        """Process heartbeat from an agent."""
        async with self._lock:
            if agent_id not in self.agents:
                return {"status": "not_found", "agent_id": agent_id}

            registration = self.agents[agent_id]
            registration.last_heartbeat = time.time()

            # Update status if provided
            if status_update:
                if "status" in status_update:
                    registration.status = AgentStatus(status_update["status"])
                if "metadata" in status_update:
                    registration.metadata.update(status_update["metadata"])

            return {
                "status": "acknowledged",
                "agent_id": agent_id,
                "timestamp": registration.last_heartbeat
            }

    async def discover(self,
                       capabilities: Optional[list[CapabilityRequirement]] = None,
                       tags: Optional[list[str]] = None,
                       status: Optional[AgentStatus] = None,
                       limit: int = 10) -> list[AgentRegistration]:
        """
        Discover agents matching criteria.
        Returns agents sorted by match score.
        """
        candidates = set(self.agents.keys())

        # Filter by capabilities
        if capabilities:
            for req in capabilities:
                key = f"{req.type.value}:{req.name}"
                if key in self.capability_index:
                    candidates &= self.capability_index[key]
                elif req.required:
                    return []  # Required capability not found

        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self.tag_index:
                    candidates &= self.tag_index[tag]

        # Filter by status
        if status:
            candidates = {
                aid for aid in candidates
                if self.agents[aid].status == status
            }

        # Score and sort candidates
        scored = []
        for agent_id in candidates:
            registration = self.agents[agent_id]
            score = self._calculate_match_score(registration, capabilities, tags)
            scored.append((score, registration))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [reg for _, reg in scored[:limit]]

    def _calculate_match_score(self,
                                registration: AgentRegistration,
                                capabilities: Optional[list[CapabilityRequirement]],
                                tags: Optional[list[str]]) -> float:
        """Calculate how well an agent matches requirements."""
        score = 1.0

        # Health bonus
        if registration.status == AgentStatus.HEALTHY:
            score += 0.2
        elif registration.status == AgentStatus.DEGRADED:
            score -= 0.1
        elif registration.status == AgentStatus.UNHEALTHY:
            score -= 0.5

        # Capability performance scores
        if capabilities:
            for req in capabilities:
                for cap in registration.capabilities:
                    if cap.matches(req):
                        score += cap.performance_score * 0.3
                        break

        # Recent activity bonus
        heartbeat_age = time.time() - registration.last_heartbeat
        if heartbeat_age < 60:
            score += 0.1
        elif heartbeat_age > 300:
            score -= 0.2

        return score

    async def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get a specific agent's registration."""
        return self.agents.get(agent_id)

    async def list_agents(self,
                          status: Optional[AgentStatus] = None,
                          tag: Optional[str] = None) -> list[AgentRegistration]:
        """List all registered agents with optional filters."""
        agents = list(self.agents.values())

        if status:
            agents = [a for a in agents if a.status == status]

        if tag:
            agents = [a for a in agents if tag in a.tags]

        return agents

    async def update_capability_score(self, agent_id: str,
                                        capability_name: str,
                                        score: float):
        """Update the performance score for an agent's capability."""
        async with self._lock:
            if agent_id in self.agents:
                for cap in self.agents[agent_id].capabilities:
                    if cap.name == capability_name:
                        cap.performance_score = max(0.0, min(1.0, score))
                        break

    # ==========================================================================
    # Health Monitoring
    # ==========================================================================

    async def start_health_monitoring(self, check_func: Optional[Callable] = None):
        """Start the health monitoring loop."""
        self._running = True
        self._check_func = check_func or self._default_health_check
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())

    async def stop_health_monitoring(self):
        """Stop the health monitoring loop."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def _health_monitor_loop(self):
        """Main health monitoring loop."""
        while self._running:
            await asyncio.sleep(1)  # Check every second which agents need health checks

            now = time.time()
            for agent_id, registration in list(self.agents.items()):
                health_state = self.health_states.get(agent_id)
                if not health_state:
                    continue

                # Check if health check is due
                interval = registration.health_check.interval_seconds
                if now - health_state.last_check >= interval:
                    asyncio.create_task(self._check_agent_health(agent_id))

    async def _check_agent_health(self, agent_id: str):
        """Perform health check for an agent."""
        if agent_id not in self.agents:
            return

        registration = self.agents[agent_id]
        health_state = self.health_states[agent_id]
        health_state.last_check = time.time()

        try:
            start = time.time()
            healthy = await asyncio.wait_for(
                self._check_func(registration),
                timeout=registration.health_check.timeout_seconds
            )
            health_state.latency_ms = (time.time() - start) * 1000

            if healthy:
                health_state.consecutive_successes += 1
                health_state.consecutive_failures = 0
                health_state.last_success = time.time()
                health_state.error_message = ""

                if health_state.consecutive_successes >= registration.health_check.healthy_threshold:
                    await self._update_health_status(agent_id, AgentStatus.HEALTHY)
            else:
                await self._handle_health_failure(agent_id, "Health check returned false")

        except asyncio.TimeoutError:
            await self._handle_health_failure(agent_id, "Health check timeout")
        except Exception as e:
            await self._handle_health_failure(agent_id, str(e))

    async def _handle_health_failure(self, agent_id: str, error: str):
        """Handle a health check failure."""
        if agent_id not in self.health_states:
            return

        health_state = self.health_states[agent_id]
        health_state.consecutive_failures += 1
        health_state.consecutive_successes = 0
        health_state.last_failure = time.time()
        health_state.error_message = error

        registration = self.agents.get(agent_id)
        if registration:
            threshold = registration.health_check.unhealthy_threshold
            if health_state.consecutive_failures >= threshold:
                await self._update_health_status(agent_id, AgentStatus.UNHEALTHY)
            elif health_state.consecutive_failures >= 1:
                await self._update_health_status(agent_id, AgentStatus.DEGRADED)

    async def _update_health_status(self, agent_id: str, status: AgentStatus):
        """Update agent health status and notify callbacks."""
        if agent_id not in self.agents:
            return

        old_status = self.agents[agent_id].status
        if old_status != status:
            self.agents[agent_id].status = status
            self.health_states[agent_id].status = status

            # Notify callbacks
            for callback in self._health_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(agent_id, old_status, status)
                    else:
                        callback(agent_id, old_status, status)
                except Exception as e:
                    logging.warning(f"Health change callback failed: {e}")

    async def _default_health_check(self, registration: AgentRegistration) -> bool:
        """Default health check - verify heartbeat recency."""
        heartbeat_age = time.time() - registration.last_heartbeat
        return heartbeat_age < registration.health_check.interval_seconds * 3

    def on_health_change(self, callback: Callable):
        """Register callback for health status changes."""
        self._health_callbacks.append(callback)

    # ==========================================================================
    # Dependency Management
    # ==========================================================================

    async def check_dependencies(self, agent_id: str) -> dict:
        """Check if all dependencies for an agent are available and healthy."""
        if agent_id not in self.agents:
            return {"status": "not_found", "agent_id": agent_id}

        registration = self.agents[agent_id]
        results = {
            "agent_id": agent_id,
            "dependencies": {},
            "all_satisfied": True
        }

        for dep_id in registration.dependencies:
            if dep_id in self.agents:
                dep = self.agents[dep_id]
                dep_status = {
                    "status": dep.status.value,
                    "healthy": dep.status == AgentStatus.HEALTHY,
                    "endpoints": [e.url for e in dep.endpoints]
                }
            else:
                dep_status = {
                    "status": "not_found",
                    "healthy": False,
                    "endpoints": []
                }
                results["all_satisfied"] = False

            results["dependencies"][dep_id] = dep_status

            if not dep_status["healthy"]:
                results["all_satisfied"] = False

        return results

    async def get_dependency_graph(self) -> dict:
        """Get the complete dependency graph of all agents."""
        graph = {}
        for agent_id, registration in self.agents.items():
            graph[agent_id] = {
                "name": registration.name,
                "status": registration.status.value,
                "dependencies": registration.dependencies,
                "dependents": []
            }

        # Calculate dependents
        for agent_id, info in graph.items():
            for dep_id in info["dependencies"]:
                if dep_id in graph:
                    graph[dep_id]["dependents"].append(agent_id)

        return graph

    # ==========================================================================
    # Statistics and Monitoring
    # ==========================================================================

    async def get_registry_stats(self) -> dict:
        """Get overall registry statistics."""
        status_counts = defaultdict(int)
        capability_counts = defaultdict(int)

        for registration in self.agents.values():
            status_counts[registration.status.value] += 1
            for cap in registration.capabilities:
                capability_counts[f"{cap.type.value}:{cap.name}"] += 1

        return {
            "total_agents": len(self.agents),
            "by_status": dict(status_counts),
            "by_capability": dict(capability_counts),
            "total_capabilities": sum(len(a.capabilities) for a in self.agents.values()),
            "total_tags": len(self.tag_index)
        }

    async def export_registry(self) -> str:
        """Export registry to JSON."""
        data = {
            "exported_at": time.time(),
            "agents": []
        }

        for registration in self.agents.values():
            agent_data = {
                "agent_id": registration.agent_id,
                "name": registration.name,
                "description": registration.description,
                "version": registration.version,
                "status": registration.status.value,
                "capabilities": [
                    {
                        "type": c.type.value,
                        "name": c.name,
                        "version": c.version,
                        "performance_score": c.performance_score
                    }
                    for c in registration.capabilities
                ],
                "endpoints": [
                    {"protocol": e.protocol, "address": e.address, "port": e.port, "path": e.path}
                    for e in registration.endpoints
                ],
                "tags": registration.tags,
                "dependencies": registration.dependencies,
                "registered_at": registration.registered_at,
                "last_heartbeat": registration.last_heartbeat
            }
            data["agents"].append(agent_data)

        return json.dumps(data, indent=2)


# =============================================================================
# Registry Client
# =============================================================================

class RegistryClient:
    """
    Client for interacting with the agent registry.
    Used by agents to register and discover other agents.
    """

    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self._agent_id: Optional[str] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

    async def register(self, registration: AgentRegistration) -> dict:
        """Register this agent with the registry."""
        result = await self.registry.register(registration)
        if result["status"] == "registered":
            self._agent_id = registration.agent_id
        return result

    async def deregister(self) -> dict:
        """Deregister this agent from the registry."""
        if self._agent_id:
            result = await self.registry.deregister(self._agent_id)
            self._agent_id = None
            return result
        return {"status": "not_registered"}

    async def start_heartbeat(self, interval: int = 30):
        """Start sending periodic heartbeats."""
        self._running = True
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(interval)
        )

    async def stop_heartbeat(self):
        """Stop the heartbeat loop."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

    async def _heartbeat_loop(self, interval: int):
        """Heartbeat loop."""
        while self._running:
            if self._agent_id:
                await self.registry.heartbeat(self._agent_id)
            await asyncio.sleep(interval)

    async def discover(self, **kwargs) -> list[AgentRegistration]:
        """Discover agents matching criteria."""
        return await self.registry.discover(**kwargs)

    async def find_by_capability(self,
                                  capability_type: CapabilityType,
                                  capability_name: str,
                                  min_version: Optional[str] = None) -> list[AgentRegistration]:
        """Find agents with a specific capability."""
        req = CapabilityRequirement(
            type=capability_type,
            name=capability_name,
            min_version=min_version
        )
        return await self.registry.discover(capabilities=[req])


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of agent registry."""
    print("=" * 60)
    print("Agent Registry Demonstration")
    print("=" * 60)

    # Create registry
    registry = AgentRegistry()

    # Create some agent registrations
    agents = [
        AgentRegistration(
            agent_id="research-agent-1",
            name="Research Agent",
            description="Performs research and information gathering",
            version="1.2.0",
            capabilities=[
                Capability(CapabilityType.SKILL, "research", "1.0.0", performance_score=0.9),
                Capability(CapabilityType.SKILL, "summarization", "1.0.0", performance_score=0.85),
                Capability(CapabilityType.TOOL, "web_search", "2.0.0", performance_score=0.95),
            ],
            endpoints=[
                AgentEndpoint("http", "localhost", 8001, "/api/v1"),
            ],
            tags=["research", "analysis"],
            max_concurrent=5
        ),
        AgentRegistration(
            agent_id="writing-agent-1",
            name="Writing Agent",
            description="Generates and edits written content",
            version="2.0.0",
            capabilities=[
                Capability(CapabilityType.SKILL, "writing", "2.0.0", performance_score=0.92),
                Capability(CapabilityType.SKILL, "editing", "1.5.0", performance_score=0.88),
                Capability(CapabilityType.DOMAIN, "technical", "1.0.0", performance_score=0.8),
            ],
            endpoints=[
                AgentEndpoint("http", "localhost", 8002, "/api/v1"),
                AgentEndpoint("grpc", "localhost", 9002),
            ],
            tags=["writing", "content"],
            dependencies=["research-agent-1"],
            max_concurrent=10
        ),
        AgentRegistration(
            agent_id="analysis-agent-1",
            name="Analysis Agent",
            description="Performs data analysis and insights",
            version="1.5.0",
            capabilities=[
                Capability(CapabilityType.SKILL, "analysis", "1.5.0", performance_score=0.87),
                Capability(CapabilityType.SKILL, "visualization", "1.0.0", performance_score=0.75),
                Capability(CapabilityType.TOOL, "python_exec", "1.0.0", performance_score=0.9),
            ],
            endpoints=[
                AgentEndpoint("http", "localhost", 8003, "/api/v1"),
            ],
            tags=["analysis", "data"],
            max_concurrent=3
        ),
    ]

    # Register agents
    print("\nRegistering agents...")
    for agent in agents:
        result = await registry.register(agent)
        print(f"  {agent.name}: {result['status']} (instance: {result['instance_id']})")

    # Send heartbeats to make agents healthy
    print("\nSending heartbeats...")
    for agent in agents:
        await registry.heartbeat(agent.agent_id, {"status": "healthy"})
        # Update status directly for demo
        registry.agents[agent.agent_id].status = AgentStatus.HEALTHY

    # Discover agents
    print("\n" + "=" * 60)
    print("Discovery Examples")
    print("=" * 60)

    # Find by capability
    print("\n1. Find agents with 'research' skill:")
    results = await registry.discover(
        capabilities=[CapabilityRequirement(CapabilityType.SKILL, "research")]
    )
    for agent in results:
        print(f"   - {agent.name} ({agent.agent_id})")

    # Find by tag
    print("\n2. Find agents with 'analysis' tag:")
    results = await registry.discover(tags=["analysis"])
    for agent in results:
        print(f"   - {agent.name} ({agent.agent_id})")

    # Find healthy agents
    print("\n3. Find healthy agents:")
    results = await registry.discover(status=AgentStatus.HEALTHY)
    for agent in results:
        print(f"   - {agent.name} ({agent.agent_id}) - Status: {agent.status.value}")

    # Check dependencies
    print("\n" + "=" * 60)
    print("Dependency Check")
    print("=" * 60)

    dep_check = await registry.check_dependencies("writing-agent-1")
    print(f"\nWriting Agent dependencies:")
    print(f"  All satisfied: {dep_check['all_satisfied']}")
    for dep_id, dep_info in dep_check['dependencies'].items():
        print(f"  - {dep_id}: {dep_info['status']}")

    # Get dependency graph
    print("\nDependency Graph:")
    graph = await registry.get_dependency_graph()
    for agent_id, info in graph.items():
        deps = info['dependencies'] or ['none']
        print(f"  {info['name']}: depends on {deps}")

    # Registry statistics
    print("\n" + "=" * 60)
    print("Registry Statistics")
    print("=" * 60)

    stats = await registry.get_registry_stats()
    print(f"\nTotal agents: {stats['total_agents']}")
    print(f"By status: {stats['by_status']}")
    print(f"Total capabilities: {stats['total_capabilities']}")

    # Use registry client
    print("\n" + "=" * 60)
    print("Registry Client Demo")
    print("=" * 60)

    client = RegistryClient(registry)

    # Register a new agent via client
    new_agent = AgentRegistration(
        agent_id="code-agent-1",
        name="Code Agent",
        description="Writes and reviews code",
        version="1.0.0",
        capabilities=[
            Capability(CapabilityType.SKILL, "coding", "1.0.0", performance_score=0.85),
            Capability(CapabilityType.TOOL, "python_exec", "1.0.0", performance_score=0.9),
        ],
        endpoints=[AgentEndpoint("http", "localhost", 8004, "/api/v1")],
        tags=["coding", "development"]
    )

    result = await client.register(new_agent)
    print(f"\nRegistered via client: {new_agent.name}")

    # Find agents with python execution capability
    print("\nFind agents with python_exec tool:")
    results = await client.find_by_capability(
        CapabilityType.TOOL,
        "python_exec"
    )
    for agent in results:
        print(f"  - {agent.name}")

    # Export registry
    print("\n" + "=" * 60)
    print("Registry Export (first 500 chars)")
    print("=" * 60)
    export = await registry.export_registry()
    print(export[:500] + "...")


if __name__ == "__main__":
    asyncio.run(main())
