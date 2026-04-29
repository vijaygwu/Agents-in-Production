"""
Chapter 19: Human-Agent Collaboration Patterns
=============================================

Implements patterns for effective human-AI collaboration:
- Human-in-the-loop (HITL) workflows
- Approval workflows
- Escalation handling
- Feedback collection
- Confidence-based routing
- Graduated autonomy

Essential patterns for responsible AI agent deployment.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict


class DecisionConfidence(Enum):
    """Confidence levels for agent decisions."""
    HIGH = "high"       # >90% - Agent can proceed autonomously
    MEDIUM = "medium"   # 70-90% - May need human review
    LOW = "low"         # <70% - Requires human intervention


class EscalationReason(Enum):
    """Reasons for escalating to humans."""
    LOW_CONFIDENCE = "low_confidence"
    HIGH_RISK = "high_risk"
    POLICY_REQUIRED = "policy_required"
    USER_REQUESTED = "user_requested"
    AMBIGUOUS_INPUT = "ambiguous_input"
    SENSITIVE_CONTENT = "sensitive_content"
    TIMEOUT = "timeout"
    ERROR = "error"


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    MODIFIED = "modified"


class FeedbackType(Enum):
    """Types of human feedback."""
    APPROVAL = "approval"
    REJECTION = "rejection"
    CORRECTION = "correction"
    RATING = "rating"
    COMMENT = "comment"
    PREFERENCE = "preference"


@dataclass
class AgentDecision:
    """A decision made by an agent that may need human review."""
    id: str
    agent_id: str
    action: str
    parameters: dict
    confidence: float
    confidence_level: DecisionConfidence
    reasoning: str
    alternatives: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_confidence_score(cls, confidence: float, **kwargs) -> 'AgentDecision':
        """Create decision with automatic confidence level."""
        if confidence >= 0.9:
            level = DecisionConfidence.HIGH
        elif confidence >= 0.7:
            level = DecisionConfidence.MEDIUM
        else:
            level = DecisionConfidence.LOW

        return cls(
            id=kwargs.pop("id", str(uuid.uuid4())[:12]),
            confidence=confidence,
            confidence_level=level,
            **kwargs
        )


@dataclass
class ApprovalRequest:
    """A request for human approval."""
    id: str
    decision: AgentDecision
    reason: EscalationReason
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    required_approvers: list[str] = field(default_factory=list)
    deadline: Optional[float] = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    responded_at: Optional[float] = None
    responder: Optional[str] = None
    response_notes: str = ""
    modified_parameters: Optional[dict] = None


@dataclass
class HumanFeedback:
    """Feedback from a human on agent behavior."""
    id: str
    feedback_type: FeedbackType
    decision_id: Optional[str] = None
    agent_id: str = ""
    rating: Optional[int] = None  # 1-5 scale
    comment: str = ""
    correction: Optional[dict] = None
    timestamp: float = field(default_factory=time.time)
    user_id: str = ""


class ConfidenceRouter:
    """
    Routes decisions based on confidence levels.
    Implements graduated autonomy patterns.
    """

    def __init__(self,
                 high_confidence_threshold: float = 0.9,
                 medium_confidence_threshold: float = 0.7):
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.action_overrides: dict[str, dict] = {}  # action -> config
        self.agent_autonomy_levels: dict[str, float] = {}  # agent_id -> autonomy

    def set_action_config(self, action: str,
                          always_require_approval: bool = False,
                          min_confidence: float = 0.7):
        """Configure routing for specific actions."""
        self.action_overrides[action] = {
            "always_require_approval": always_require_approval,
            "min_confidence": min_confidence
        }

    def set_agent_autonomy(self, agent_id: str, autonomy_level: float):
        """
        Set autonomy level for an agent (0.0 to 1.0).
        Higher autonomy = more decisions can be made without human review.
        """
        self.agent_autonomy_levels[agent_id] = max(0.0, min(1.0, autonomy_level))

    def route(self, decision: AgentDecision) -> tuple[bool, str]:
        """
        Determine if decision needs human approval.

        Returns:
            (needs_approval, reason)
        """
        # Check action-specific overrides
        if decision.action in self.action_overrides:
            config = self.action_overrides[decision.action]
            if config["always_require_approval"]:
                return True, "Action requires approval"
            if decision.confidence < config["min_confidence"]:
                return True, f"Confidence below minimum for {decision.action}"

        # Check agent autonomy level
        autonomy = self.agent_autonomy_levels.get(decision.agent_id, 0.5)

        # Adjust thresholds based on autonomy
        effective_high = self.high_threshold * (1 - autonomy * 0.2)
        effective_medium = self.medium_threshold * (1 - autonomy * 0.2)

        if decision.confidence >= effective_high:
            return False, "High confidence - autonomous execution"
        elif decision.confidence >= effective_medium:
            # Medium confidence - probabilistic approval based on autonomy
            if autonomy >= 0.8:
                return False, "Medium confidence but high autonomy"
            return True, "Medium confidence - human review recommended"
        else:
            return True, "Low confidence - human approval required"


class ApprovalWorkflow:
    """
    Manages approval workflows for agent decisions.
    """

    def __init__(self,
                 default_timeout: int = 3600,  # 1 hour
                 auto_approve_on_timeout: bool = False):
        self.default_timeout = default_timeout
        self.auto_approve_on_timeout = auto_approve_on_timeout
        self.pending_requests: dict[str, ApprovalRequest] = {}
        self.completed_requests: list[ApprovalRequest] = []
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def create_request(self,
                              decision: AgentDecision,
                              reason: EscalationReason,
                              priority: int = 1,
                              required_approvers: Optional[list[str]] = None,
                              timeout: Optional[int] = None) -> ApprovalRequest:
        """Create an approval request."""
        request = ApprovalRequest(
            id=f"approval_{int(time.time()*1000)}",
            decision=decision,
            reason=reason,
            priority=priority,
            required_approvers=required_approvers or [],
            deadline=time.time() + (timeout or self.default_timeout)
        )

        async with self._lock:
            self.pending_requests[request.id] = request

        # Notify listeners
        await self._notify("created", request)

        return request

    async def approve(self,
                      request_id: str,
                      approver: str,
                      notes: str = "",
                      modified_parameters: Optional[dict] = None) -> bool:
        """Approve a request."""
        async with self._lock:
            if request_id not in self.pending_requests:
                return False

            request = self.pending_requests[request_id]

            # Check if approver is authorized
            if request.required_approvers and approver not in request.required_approvers:
                return False

            request.status = ApprovalStatus.MODIFIED if modified_parameters else ApprovalStatus.APPROVED
            request.responded_at = time.time()
            request.responder = approver
            request.response_notes = notes
            request.modified_parameters = modified_parameters

            self.completed_requests.append(request)
            del self.pending_requests[request_id]

        await self._notify("approved", request)
        return True

    async def reject(self,
                     request_id: str,
                     rejector: str,
                     reason: str = "") -> bool:
        """Reject a request."""
        async with self._lock:
            if request_id not in self.pending_requests:
                return False

            request = self.pending_requests[request_id]
            request.status = ApprovalStatus.REJECTED
            request.responded_at = time.time()
            request.responder = rejector
            request.response_notes = reason

            self.completed_requests.append(request)
            del self.pending_requests[request_id]

        await self._notify("rejected", request)
        return True

    async def check_timeouts(self):
        """Check for timed-out requests."""
        now = time.time()
        expired = []

        async with self._lock:
            for request_id, request in list(self.pending_requests.items()):
                if request.deadline and now > request.deadline:
                    if self.auto_approve_on_timeout:
                        request.status = ApprovalStatus.APPROVED
                        request.response_notes = "Auto-approved on timeout"
                    else:
                        request.status = ApprovalStatus.EXPIRED

                    request.responded_at = now
                    self.completed_requests.append(request)
                    expired.append(request)
                    del self.pending_requests[request_id]

        for request in expired:
            await self._notify("timeout", request)

    async def wait_for_approval(self,
                                 request_id: str,
                                 poll_interval: float = 1.0) -> ApprovalRequest:
        """Wait for a request to be processed."""
        while True:
            await self.check_timeouts()

            async with self._lock:
                if request_id not in self.pending_requests:
                    # Find in completed
                    for req in reversed(self.completed_requests):
                        if req.id == request_id:
                            return req
                    raise ValueError(f"Request {request_id} not found")

            await asyncio.sleep(poll_interval)

    def on_event(self, event: str, callback: Callable):
        """Register callback for workflow events."""
        self._callbacks[event].append(callback)

    async def _notify(self, event: str, request: ApprovalRequest):
        """Notify callbacks of an event."""
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(request)
                else:
                    callback(request)
            except Exception:
                pass

    def get_pending(self, approver: Optional[str] = None) -> list[ApprovalRequest]:
        """Get pending approval requests."""
        requests = list(self.pending_requests.values())
        if approver:
            requests = [
                r for r in requests
                if not r.required_approvers or approver in r.required_approvers
            ]
        return sorted(requests, key=lambda r: (r.priority, r.created_at), reverse=True)


class EscalationManager:
    """
    Manages escalation paths for agent decisions.
    """

    def __init__(self):
        self.escalation_rules: list[dict] = []
        self.escalation_handlers: dict[str, Callable] = {}
        self.escalation_history: list[dict] = []

    def add_rule(self,
                 condition: Callable[[AgentDecision], bool],
                 handler: str,
                 priority: int = 1):
        """Add an escalation rule."""
        self.escalation_rules.append({
            "condition": condition,
            "handler": handler,
            "priority": priority
        })
        self.escalation_rules.sort(key=lambda r: r["priority"], reverse=True)

    def register_handler(self, name: str, handler: Callable):
        """Register an escalation handler."""
        self.escalation_handlers[name] = handler

    async def escalate(self, decision: AgentDecision,
                        reason: EscalationReason) -> dict:
        """
        Escalate a decision based on rules.
        Returns escalation result.
        """
        # Find matching handler
        handler_name = None
        for rule in self.escalation_rules:
            if rule["condition"](decision):
                handler_name = rule["handler"]
                break

        if not handler_name:
            handler_name = "default"

        if handler_name not in self.escalation_handlers:
            return {"error": f"No handler for {handler_name}"}

        # Execute handler
        handler = self.escalation_handlers[handler_name]

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(decision, reason)
            else:
                result = handler(decision, reason)
        except Exception as e:
            result = {"error": str(e)}

        # Record escalation
        self.escalation_history.append({
            "decision_id": decision.id,
            "reason": reason.value,
            "handler": handler_name,
            "result": result,
            "timestamp": time.time()
        })

        return result


class FeedbackCollector:
    """
    Collects and manages human feedback on agent behavior.
    """

    def __init__(self):
        self.feedback: list[HumanFeedback] = []
        self.feedback_by_decision: dict[str, list[HumanFeedback]] = defaultdict(list)
        self.feedback_by_agent: dict[str, list[HumanFeedback]] = defaultdict(list)
        self._callbacks: list[Callable] = []

    async def submit(self, feedback: HumanFeedback):
        """Submit feedback."""
        self.feedback.append(feedback)

        if feedback.decision_id:
            self.feedback_by_decision[feedback.decision_id].append(feedback)
        if feedback.agent_id:
            self.feedback_by_agent[feedback.agent_id].append(feedback)

        # Notify listeners
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(feedback)
                else:
                    callback(feedback)
            except Exception:
                pass

    def on_feedback(self, callback: Callable):
        """Register callback for new feedback."""
        self._callbacks.append(callback)

    def get_agent_stats(self, agent_id: str) -> dict:
        """Get feedback statistics for an agent."""
        agent_feedback = self.feedback_by_agent.get(agent_id, [])

        if not agent_feedback:
            return {"agent_id": agent_id, "feedback_count": 0}

        ratings = [f.rating for f in agent_feedback if f.rating is not None]
        by_type = defaultdict(int)
        for f in agent_feedback:
            by_type[f.feedback_type.value] += 1

        return {
            "agent_id": agent_id,
            "feedback_count": len(agent_feedback),
            "average_rating": sum(ratings) / len(ratings) if ratings else None,
            "by_type": dict(by_type),
            "approval_rate": (
                by_type.get("approval", 0) /
                (by_type.get("approval", 0) + by_type.get("rejection", 0))
                if (by_type.get("approval", 0) + by_type.get("rejection", 0)) > 0
                else None
            )
        }


class HITLWorkflow:
    """
    Human-in-the-Loop workflow orchestrator.
    Coordinates all HITL components.
    """

    def __init__(self):
        self.router = ConfidenceRouter()
        self.approval_workflow = ApprovalWorkflow()
        self.escalation_manager = EscalationManager()
        self.feedback_collector = FeedbackCollector()
        self._execution_handlers: dict[str, Callable] = {}

    def register_executor(self, action: str, handler: Callable):
        """Register handler for executing approved actions."""
        self._execution_handlers[action] = handler

    async def process_decision(self,
                                decision: AgentDecision,
                                auto_execute: bool = True) -> dict:
        """
        Process an agent decision through the HITL workflow.

        Returns execution result or approval request.
        """
        # Route decision
        needs_approval, reason = self.router.route(decision)

        if not needs_approval:
            # Execute autonomously
            if auto_execute and decision.action in self._execution_handlers:
                result = await self._execute(decision)
                return {
                    "status": "executed",
                    "approval_required": False,
                    "result": result
                }
            return {
                "status": "ready",
                "approval_required": False,
                "decision": decision
            }

        # Create approval request
        escalation_reason = self._determine_escalation_reason(decision, reason)
        request = await self.approval_workflow.create_request(
            decision=decision,
            reason=escalation_reason,
            priority=self._determine_priority(decision)
        )

        return {
            "status": "pending_approval",
            "approval_required": True,
            "request_id": request.id,
            "reason": reason
        }

    async def approve_and_execute(self,
                                   request_id: str,
                                   approver: str,
                                   modified_parameters: Optional[dict] = None) -> dict:
        """Approve a request and execute the action."""
        success = await self.approval_workflow.approve(
            request_id, approver,
            modified_parameters=modified_parameters
        )

        if not success:
            return {"error": "Approval failed"}

        # Find the request
        for request in self.approval_workflow.completed_requests:
            if request.id == request_id:
                decision = request.decision

                # Apply modifications if any
                if request.modified_parameters:
                    decision.parameters.update(request.modified_parameters)

                # Execute
                if decision.action in self._execution_handlers:
                    result = await self._execute(decision)
                    return {
                        "status": "executed",
                        "result": result,
                        "modified": bool(request.modified_parameters)
                    }

                return {"status": "approved", "decision": decision}

        return {"error": "Request not found"}

    async def _execute(self, decision: AgentDecision) -> Any:
        """Execute a decision."""
        handler = self._execution_handlers.get(decision.action)
        if not handler:
            raise ValueError(f"No handler for action: {decision.action}")

        if asyncio.iscoroutinefunction(handler):
            return await handler(decision.parameters)
        return handler(decision.parameters)

    def _determine_escalation_reason(self, decision: AgentDecision,
                                      routing_reason: str) -> EscalationReason:
        """Determine escalation reason from decision."""
        if decision.confidence_level == DecisionConfidence.LOW:
            return EscalationReason.LOW_CONFIDENCE
        if "risk" in routing_reason.lower():
            return EscalationReason.HIGH_RISK
        if "policy" in routing_reason.lower():
            return EscalationReason.POLICY_REQUIRED
        return EscalationReason.LOW_CONFIDENCE

    def _determine_priority(self, decision: AgentDecision) -> int:
        """Determine priority from decision."""
        if decision.context.get("urgent"):
            return 3
        if decision.confidence_level == DecisionConfidence.LOW:
            return 2
        return 1


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of human-agent collaboration patterns."""
    print("=" * 60)
    print("Human-Agent Collaboration Demonstration")
    print("=" * 60)

    # Create HITL workflow
    hitl = HITLWorkflow()

    # Configure router
    hitl.router.set_action_config("delete", always_require_approval=True)
    hitl.router.set_action_config("transfer_funds", min_confidence=0.95)
    hitl.router.set_agent_autonomy("trusted-agent", 0.9)
    hitl.router.set_agent_autonomy("new-agent", 0.3)

    # Register execution handlers
    async def execute_send_email(params: dict) -> dict:
        print(f"    [EXECUTING] Sending email to {params.get('recipient')}")
        return {"sent": True, "recipient": params.get("recipient")}

    async def execute_create_report(params: dict) -> dict:
        print(f"    [EXECUTING] Creating report: {params.get('title')}")
        return {"created": True, "report_id": "R-123"}

    hitl.register_executor("send_email", execute_send_email)
    hitl.register_executor("create_report", execute_create_report)

    # Set up callbacks
    async def on_approval_created(request: ApprovalRequest):
        print(f"\n    [APPROVAL NEEDED] {request.id}")
        print(f"      Action: {request.decision.action}")
        print(f"      Reason: {request.reason.value}")
        print(f"      Confidence: {request.decision.confidence:.2f}")

    hitl.approval_workflow.on_event("created", on_approval_created)

    # Test scenarios
    print("\n" + "-" * 40)
    print("Scenario 1: High confidence autonomous decision")
    print("-" * 40)

    decision1 = AgentDecision.from_confidence_score(
        confidence=0.95,
        agent_id="trusted-agent",
        action="send_email",
        parameters={"recipient": "user@example.com", "subject": "Report"},
        reasoning="Standard notification email"
    )

    result1 = await hitl.process_decision(decision1)
    print(f"Result: {result1['status']}")

    print("\n" + "-" * 40)
    print("Scenario 2: Medium confidence needs review")
    print("-" * 40)

    decision2 = AgentDecision.from_confidence_score(
        confidence=0.75,
        agent_id="new-agent",
        action="create_report",
        parameters={"title": "Q4 Analysis", "type": "financial"},
        reasoning="User requested quarterly report"
    )

    result2 = await hitl.process_decision(decision2)
    print(f"Result: {result2['status']}")

    # Simulate approval
    if result2["approval_required"]:
        print("\n  [HUMAN] Approving the report creation...")
        exec_result = await hitl.approve_and_execute(
            result2["request_id"],
            approver="admin@example.com"
        )
        print(f"  Execution result: {exec_result}")

    print("\n" + "-" * 40)
    print("Scenario 3: Action requiring approval")
    print("-" * 40)

    decision3 = AgentDecision.from_confidence_score(
        confidence=0.99,  # Even high confidence needs approval
        agent_id="trusted-agent",
        action="delete",
        parameters={"resource": "old-data", "permanent": True},
        reasoning="User requested deletion"
    )

    result3 = await hitl.process_decision(decision3)
    print(f"Result: {result3['status']}")
    if result3.get("reason"):
        print(f"Reason: {result3['reason']}")

    # Reject this one
    if result3["approval_required"]:
        print("\n  [HUMAN] Rejecting the deletion...")
        await hitl.approval_workflow.reject(
            result3["request_id"],
            rejector="admin@example.com",
            reason="Need more context before deletion"
        )

    print("\n" + "-" * 40)
    print("Scenario 4: Low confidence decision")
    print("-" * 40)

    decision4 = AgentDecision.from_confidence_score(
        confidence=0.45,
        agent_id="new-agent",
        action="send_email",
        parameters={"recipient": "vip@example.com", "subject": "Urgent"},
        reasoning="Unsure about the recipient",
        alternatives=[
            {"recipient": "support@example.com"},
            {"recipient": "team@example.com"}
        ]
    )

    result4 = await hitl.process_decision(decision4)
    print(f"Result: {result4['status']}")

    # Approve with modifications
    if result4["approval_required"]:
        print("\n  [HUMAN] Approving with modified recipient...")
        exec_result = await hitl.approve_and_execute(
            result4["request_id"],
            approver="admin@example.com",
            modified_parameters={"recipient": "team@example.com"}
        )
        print(f"  Execution result: {exec_result}")

    # Feedback collection
    print("\n" + "-" * 40)
    print("Collecting feedback")
    print("-" * 40)

    await hitl.feedback_collector.submit(HumanFeedback(
        id="fb-1",
        feedback_type=FeedbackType.APPROVAL,
        decision_id=decision1.id,
        agent_id="trusted-agent",
        rating=5,
        comment="Good decision",
        user_id="admin"
    ))

    await hitl.feedback_collector.submit(HumanFeedback(
        id="fb-2",
        feedback_type=FeedbackType.CORRECTION,
        decision_id=decision4.id,
        agent_id="new-agent",
        rating=3,
        correction={"recipient": "team@example.com"},
        comment="Should have used team email",
        user_id="admin"
    ))

    # Get feedback stats
    stats = hitl.feedback_collector.get_agent_stats("trusted-agent")
    print(f"\nTrusted Agent feedback: {stats}")

    stats = hitl.feedback_collector.get_agent_stats("new-agent")
    print(f"New Agent feedback: {stats}")

    # Summary
    print("\n" + "=" * 60)
    print("Workflow Summary")
    print("=" * 60)

    pending = hitl.approval_workflow.get_pending()
    completed = hitl.approval_workflow.completed_requests

    print(f"\nPending approvals: {len(pending)}")
    print(f"Completed approvals: {len(completed)}")

    for req in completed:
        print(f"  - {req.decision.action}: {req.status.value}")


if __name__ == "__main__":
    asyncio.run(main())
