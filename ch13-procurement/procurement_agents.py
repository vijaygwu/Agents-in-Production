"""
Chapter 14: Case Study - Enterprise Procurement System
======================================================

A production-ready procurement system demonstrating:
- Multi-agent orchestration
- Human-in-the-loop for high-value approvals
- Policy enforcement
- Cost tracking
- Full observability

This system handles purchase requests from intake through approval.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable
from collections import defaultdict


# =============================================================================
# Domain Models
# =============================================================================

class PurchaseCategory(Enum):
    """Categories of purchases."""
    SOFTWARE = "software"
    HARDWARE = "hardware"
    SERVICES = "services"
    OFFICE_SUPPLIES = "office_supplies"
    TRAVEL = "travel"


class ApprovalLevel(Enum):
    """Approval levels based on amount."""
    MANAGER = "manager"      # Up to $5,000
    DIRECTOR = "director"    # $5,001 - $25,000
    VP = "vp"               # $25,001 - $100,000
    EXECUTIVE = "executive"  # Over $100,000


class RequestStatus(Enum):
    """Status of a purchase request."""
    DRAFT = "draft"
    PENDING_ANALYSIS = "pending_analysis"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ORDERED = "ordered"
    COMPLETED = "completed"


@dataclass
class Vendor:
    """A vendor/supplier."""
    id: str
    name: str
    category: PurchaseCategory
    rating: float = 0.0  # 0-5
    compliance_certified: bool = False
    preferred: bool = False
    contact_email: str = ""


@dataclass
class PurchaseItem:
    """An item in a purchase request."""
    name: str
    category: PurchaseCategory
    quantity: int
    unit_price: float
    vendor_id: Optional[str] = None
    justification: str = ""

    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price


@dataclass
class PurchaseRequest:
    """A purchase request to be processed."""
    id: str
    requester_id: str
    requester_department: str
    items: list[PurchaseItem]
    justification: str
    urgency: str = "normal"  # low, normal, high, urgent
    status: RequestStatus = RequestStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    assigned_agent: Optional[str] = None
    analysis_result: Optional[dict] = None
    approval_chain: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def total_amount(self) -> float:
        return sum(item.total_price for item in self.items)

    @property
    def required_approval_level(self) -> ApprovalLevel:
        amount = self.total_amount
        if amount <= 5000:
            return ApprovalLevel.MANAGER
        elif amount <= 25000:
            return ApprovalLevel.DIRECTOR
        elif amount <= 100000:
            return ApprovalLevel.VP
        else:
            return ApprovalLevel.EXECUTIVE


# =============================================================================
# Agent Definitions
# =============================================================================

class ProcurementAgent:
    """Base class for procurement agents."""

    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.processed_count = 0
        self.error_count = 0

    async def process(self, request: PurchaseRequest, context: dict) -> dict:
        """Process a request. Override in subclasses."""
        raise NotImplementedError


class IntakeAgent(ProcurementAgent):
    """
    Handles initial intake and validation of purchase requests.
    """

    def __init__(self):
        super().__init__("intake-agent", "Intake Agent")
        self.validation_rules = []

    def add_validation_rule(self, rule: Callable[[PurchaseRequest], tuple[bool, str]]):
        """Add a validation rule."""
        self.validation_rules.append(rule)

    async def process(self, request: PurchaseRequest, context: dict) -> dict:
        """Validate and prepare request for processing."""
        self.processed_count += 1
        errors = []
        warnings = []

        # Basic validation
        if not request.items:
            errors.append("Request must have at least one item")

        if not request.justification:
            warnings.append("No justification provided")

        if request.total_amount <= 0:
            errors.append("Total amount must be positive")

        # Check for duplicate requests (simplified)
        if context.get("recent_requests"):
            for recent in context["recent_requests"]:
                if (recent.requester_id == request.requester_id and
                    recent.total_amount == request.total_amount and
                    time.time() - recent.created_at < 86400):  # 24 hours
                    warnings.append("Similar request submitted recently")

        # Apply custom rules
        for rule in self.validation_rules:
            valid, message = rule(request)
            if not valid:
                errors.append(message)

        # Categorize items
        categories = set(item.category for item in request.items)

        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_amount": request.total_amount,
            "categories": [c.value for c in categories],
            "requires_approval_level": request.required_approval_level.value,
            "processed_at": time.time()
        }

        if result["valid"]:
            request.status = RequestStatus.PENDING_ANALYSIS

        return result


class AnalysisAgent(ProcurementAgent):
    """
    Analyzes purchase requests for cost optimization and compliance.
    """

    def __init__(self, vendor_database: dict[str, Vendor]):
        super().__init__("analysis-agent", "Analysis Agent")
        self.vendor_database = vendor_database
        self.compliance_rules = {}

    async def process(self, request: PurchaseRequest, context: dict) -> dict:
        """Analyze request for optimization opportunities."""
        self.processed_count += 1

        analysis = {
            "vendor_recommendations": [],
            "cost_savings": 0.0,
            "compliance_issues": [],
            "risk_assessment": "low",
            "recommendations": []
        }

        # Analyze each item
        for item in request.items:
            item_analysis = await self._analyze_item(item)
            analysis["vendor_recommendations"].extend(
                item_analysis.get("vendors", [])
            )
            analysis["cost_savings"] += item_analysis.get("potential_savings", 0)

            if item_analysis.get("compliance_flag"):
                analysis["compliance_issues"].append({
                    "item": item.name,
                    "issue": item_analysis["compliance_flag"]
                })

        # Overall risk assessment
        if request.total_amount > 50000:
            analysis["risk_assessment"] = "medium"
        if request.total_amount > 100000 or len(analysis["compliance_issues"]) > 0:
            analysis["risk_assessment"] = "high"

        # Generate recommendations
        if analysis["cost_savings"] > request.total_amount * 0.1:
            analysis["recommendations"].append(
                f"Potential cost savings of ${analysis['cost_savings']:.2f} identified"
            )

        if request.urgency == "urgent" and analysis["risk_assessment"] == "high":
            analysis["recommendations"].append(
                "High-value urgent request - recommend expedited review"
            )

        # Store analysis result
        request.analysis_result = analysis
        request.status = RequestStatus.PENDING_APPROVAL

        return analysis

    async def _analyze_item(self, item: PurchaseItem) -> dict:
        """Analyze a single item."""
        result = {"vendors": [], "potential_savings": 0, "compliance_flag": None}

        # Find alternative vendors
        matching_vendors = [
            v for v in self.vendor_database.values()
            if v.category == item.category
        ]

        # Prefer preferred vendors
        preferred = [v for v in matching_vendors if v.preferred]
        if preferred:
            best = max(preferred, key=lambda v: v.rating)
            result["vendors"].append({
                "vendor_id": best.id,
                "vendor_name": best.name,
                "rating": best.rating,
                "preferred": True
            })

            # Estimate savings from preferred vendor
            result["potential_savings"] = item.total_price * 0.05  # 5% estimated

        # Check compliance
        if item.total_price > 10000:
            non_certified = [v for v in matching_vendors if not v.compliance_certified]
            if item.vendor_id in [v.id for v in non_certified]:
                result["compliance_flag"] = "Vendor not compliance certified for amounts over $10,000"

        return result


class ApprovalAgent(ProcurementAgent):
    """
    Manages the approval workflow for purchase requests.
    """

    def __init__(self):
        super().__init__("approval-agent", "Approval Agent")
        self.approvers = {}  # level -> list of approver IDs
        self.approval_thresholds = {
            ApprovalLevel.MANAGER: 5000,
            ApprovalLevel.DIRECTOR: 25000,
            ApprovalLevel.VP: 100000,
            ApprovalLevel.EXECUTIVE: float('inf')
        }

    def register_approvers(self, level: ApprovalLevel, approvers: list[str]):
        """Register approvers for a level."""
        self.approvers[level] = approvers

    async def process(self, request: PurchaseRequest, context: dict) -> dict:
        """Process approval workflow."""
        self.processed_count += 1

        required_level = request.required_approval_level

        # Check if auto-approval is possible
        auto_approve = await self._check_auto_approval(request)

        if auto_approve["eligible"]:
            request.approval_chain.append({
                "level": "auto",
                "approver": "system",
                "decision": "approved",
                "timestamp": time.time(),
                "reason": auto_approve["reason"]
            })
            request.status = RequestStatus.APPROVED
            return {
                "status": "auto_approved",
                "reason": auto_approve["reason"]
            }

        # Build approval chain
        chain = []
        for level in ApprovalLevel:
            if self.approval_thresholds[level] >= request.total_amount:
                chain.append({
                    "level": level.value,
                    "approvers": self.approvers.get(level, []),
                    "required": True,
                    "status": "pending"
                })
                break
            elif level in self.approvers:
                chain.append({
                    "level": level.value,
                    "approvers": self.approvers.get(level, []),
                    "required": True,
                    "status": "pending"
                })

        return {
            "status": "pending_approval",
            "required_level": required_level.value,
            "approval_chain": chain,
            "human_review_required": request.total_amount > 10000 or
                                    (request.analysis_result or {}).get("risk_assessment") == "high"
        }

    async def _check_auto_approval(self, request: PurchaseRequest) -> dict:
        """Check if request can be auto-approved."""
        # Auto-approve low-value office supplies from known vendors
        if (request.total_amount <= 500 and
            all(item.category == PurchaseCategory.OFFICE_SUPPLIES for item in request.items)):
            return {
                "eligible": True,
                "reason": "Low-value office supplies auto-approved"
            }

        # Auto-approve if it's a reorder of previously approved items
        if request.notes and "reorder" in " ".join(request.notes).lower():
            if request.total_amount <= 1000:
                return {
                    "eligible": True,
                    "reason": "Low-value reorder auto-approved"
                }

        return {"eligible": False, "reason": None}

    async def submit_approval(self, request: PurchaseRequest,
                               approver: str, approved: bool, notes: str = "") -> dict:
        """Submit an approval decision."""
        request.approval_chain.append({
            "approver": approver,
            "decision": "approved" if approved else "rejected",
            "timestamp": time.time(),
            "notes": notes
        })

        if approved:
            request.status = RequestStatus.APPROVED
        else:
            request.status = RequestStatus.REJECTED

        request.updated_at = time.time()

        return {
            "status": request.status.value,
            "approver": approver,
            "decision": "approved" if approved else "rejected"
        }


class FulfillmentAgent(ProcurementAgent):
    """
    Handles order fulfillment after approval.
    """

    def __init__(self):
        super().__init__("fulfillment-agent", "Fulfillment Agent")
        self.orders = []

    async def process(self, request: PurchaseRequest, context: dict) -> dict:
        """Process approved request for fulfillment."""
        self.processed_count += 1

        if request.status != RequestStatus.APPROVED:
            return {"error": "Request not approved"}

        # Create order
        order = {
            "order_id": f"PO-{int(time.time())}-{len(self.orders)}",
            "request_id": request.id,
            "items": [
                {
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "vendor_id": item.vendor_id
                }
                for item in request.items
            ],
            "total_amount": request.total_amount,
            "created_at": time.time(),
            "status": "pending"
        }

        self.orders.append(order)
        request.status = RequestStatus.ORDERED

        return {
            "status": "ordered",
            "order_id": order["order_id"],
            "estimated_delivery": "5-7 business days"  # Simplified
        }


# =============================================================================
# Orchestrator
# =============================================================================

class ProcurementOrchestrator:
    """
    Orchestrates the multi-agent procurement workflow.
    """

    def __init__(self):
        self.intake = IntakeAgent()
        self.analysis = AnalysisAgent({})
        self.approval = ApprovalAgent()
        self.fulfillment = FulfillmentAgent()

        self.requests: dict[str, PurchaseRequest] = {}
        self.metrics = defaultdict(int)
        self._event_handlers: dict[str, list[Callable]] = defaultdict(list)

    def set_vendor_database(self, vendors: dict[str, Vendor]):
        """Set the vendor database."""
        self.analysis.vendor_database = vendors

    def on_event(self, event: str, handler: Callable):
        """Register event handler."""
        self._event_handlers[event].append(handler)

    async def _emit(self, event: str, data: Any):
        """Emit an event."""
        for handler in self._event_handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception:
                pass

    async def submit_request(self, request: PurchaseRequest) -> dict:
        """Submit a new purchase request."""
        self.requests[request.id] = request
        self.metrics["requests_submitted"] += 1

        await self._emit("request_submitted", request)

        # Process through intake
        context = {"recent_requests": list(self.requests.values())[-10:]}
        intake_result = await self.intake.process(request, context)

        if not intake_result["valid"]:
            self.metrics["requests_rejected_intake"] += 1
            return {
                "status": "validation_failed",
                "errors": intake_result["errors"]
            }

        await self._emit("request_validated", request)

        # Process through analysis
        analysis_result = await self.analysis.process(request, context)

        await self._emit("request_analyzed", request)

        # Process through approval workflow
        approval_result = await self.approval.process(request, context)

        if approval_result["status"] == "auto_approved":
            self.metrics["requests_auto_approved"] += 1
            await self._emit("request_approved", request)

            # Automatic fulfillment for auto-approved
            fulfillment_result = await self.fulfillment.process(request, context)
            await self._emit("request_fulfilled", request)

            return {
                "status": "completed",
                "request_id": request.id,
                "auto_approved": True,
                "order": fulfillment_result
            }

        self.metrics["requests_pending_approval"] += 1
        return {
            "status": "pending_approval",
            "request_id": request.id,
            "approval_chain": approval_result["approval_chain"],
            "human_review_required": approval_result.get("human_review_required", False)
        }

    async def approve_request(self, request_id: str, approver: str,
                               approved: bool, notes: str = "") -> dict:
        """Process an approval decision."""
        if request_id not in self.requests:
            return {"error": "Request not found"}

        request = self.requests[request_id]

        approval_result = await self.approval.submit_approval(
            request, approver, approved, notes
        )

        if approved:
            self.metrics["requests_approved"] += 1
            await self._emit("request_approved", request)

            # Process fulfillment
            fulfillment_result = await self.fulfillment.process(request, {})
            await self._emit("request_fulfilled", request)

            return {
                "status": "fulfilled",
                "approval": approval_result,
                "order": fulfillment_result
            }
        else:
            self.metrics["requests_rejected"] += 1
            await self._emit("request_rejected", request)

            return {
                "status": "rejected",
                "approval": approval_result
            }

    def get_pending_approvals(self) -> list[dict]:
        """Get all pending approval requests."""
        pending = []
        for request in self.requests.values():
            if request.status == RequestStatus.PENDING_APPROVAL:
                pending.append({
                    "request_id": request.id,
                    "requester": request.requester_id,
                    "department": request.requester_department,
                    "amount": request.total_amount,
                    "urgency": request.urgency,
                    "analysis": request.analysis_result,
                    "created_at": request.created_at
                })
        return sorted(pending, key=lambda x: (
            {"urgent": 0, "high": 1, "normal": 2, "low": 3}[x["urgency"]],
            x["created_at"]
        ))

    def get_metrics(self) -> dict:
        """Get system metrics."""
        return {
            **dict(self.metrics),
            "agents": {
                "intake": {"processed": self.intake.processed_count},
                "analysis": {"processed": self.analysis.processed_count},
                "approval": {"processed": self.approval.processed_count},
                "fulfillment": {"processed": self.fulfillment.processed_count}
            }
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of procurement system."""
    print("=" * 60)
    print("Multi-Agent Procurement System")
    print("=" * 60)

    # Initialize system
    orchestrator = ProcurementOrchestrator()

    # Set up vendors
    vendors = {
        "v1": Vendor("v1", "Office Depot", PurchaseCategory.OFFICE_SUPPLIES,
                    4.5, True, True, "sales@officedepot.com"),
        "v2": Vendor("v2", "Dell Technologies", PurchaseCategory.HARDWARE,
                    4.8, True, True, "enterprise@dell.com"),
        "v3": Vendor("v3", "Acme Software", PurchaseCategory.SOFTWARE,
                    3.9, False, False, "sales@acme.com"),
        "v4": Vendor("v4", "AWS", PurchaseCategory.SOFTWARE,
                    4.9, True, True, "enterprise@aws.com")
    }
    orchestrator.set_vendor_database(vendors)

    # Set up approvers
    orchestrator.approval.register_approvers(ApprovalLevel.MANAGER, ["manager@company.com"])
    orchestrator.approval.register_approvers(ApprovalLevel.DIRECTOR, ["director@company.com"])
    orchestrator.approval.register_approvers(ApprovalLevel.VP, ["vp@company.com"])

    # Event handlers
    async def on_request(request):
        print(f"\n  [EVENT] Request {request.id}: {request.status.value}")

    orchestrator.on_event("request_submitted", on_request)
    orchestrator.on_event("request_approved", on_request)

    print("\n" + "-" * 40)
    print("Scenario 1: Low-value auto-approved request")
    print("-" * 40)

    request1 = PurchaseRequest(
        id="REQ-001",
        requester_id="emp123",
        requester_department="Engineering",
        items=[
            PurchaseItem("Pens", PurchaseCategory.OFFICE_SUPPLIES, 50, 2.00, "v1"),
            PurchaseItem("Notebooks", PurchaseCategory.OFFICE_SUPPLIES, 20, 5.00, "v1")
        ],
        justification="Regular office supplies",
        urgency="low"
    )

    result1 = await orchestrator.submit_request(request1)
    print(f"\nResult: {result1['status']}")
    if result1.get('auto_approved'):
        print(f"Order ID: {result1['order']['order_id']}")

    print("\n" + "-" * 40)
    print("Scenario 2: Medium-value request requiring approval")
    print("-" * 40)

    request2 = PurchaseRequest(
        id="REQ-002",
        requester_id="emp456",
        requester_department="IT",
        items=[
            PurchaseItem("Laptop", PurchaseCategory.HARDWARE, 5, 1500.00, "v2"),
            PurchaseItem("Monitors", PurchaseCategory.HARDWARE, 10, 300.00, "v2")
        ],
        justification="New hire equipment",
        urgency="high"
    )

    result2 = await orchestrator.submit_request(request2)
    print(f"\nResult: {result2['status']}")
    print(f"Total Amount: ${request2.total_amount:,.2f}")
    print(f"Requires: {request2.required_approval_level.value} approval")

    # Get pending approvals
    pending = orchestrator.get_pending_approvals()
    print(f"\nPending approvals: {len(pending)}")
    for p in pending:
        print(f"  - {p['request_id']}: ${p['amount']:,.2f} ({p['urgency']})")

    # Approve request
    print("\n  [HUMAN] Approving request REQ-002...")
    approval_result = await orchestrator.approve_request(
        "REQ-002",
        "manager@company.com",
        approved=True,
        notes="Approved for new hires"
    )
    print(f"  Result: {approval_result['status']}")

    print("\n" + "-" * 40)
    print("Scenario 3: High-value request with compliance concerns")
    print("-" * 40)

    request3 = PurchaseRequest(
        id="REQ-003",
        requester_id="emp789",
        requester_department="Marketing",
        items=[
            PurchaseItem("Enterprise CRM", PurchaseCategory.SOFTWARE, 1, 75000.00, "v3"),
        ],
        justification="Critical marketing automation platform",
        urgency="urgent"
    )

    result3 = await orchestrator.submit_request(request3)
    print(f"\nResult: {result3['status']}")
    print(f"Total Amount: ${request3.total_amount:,.2f}")
    print(f"Analysis: {request3.analysis_result.get('risk_assessment', 'N/A')}")

    if request3.analysis_result.get("compliance_issues"):
        print("Compliance issues:")
        for issue in request3.analysis_result["compliance_issues"]:
            print(f"  - {issue['item']}: {issue['issue']}")

    # Show metrics
    print("\n" + "=" * 60)
    print("System Metrics")
    print("=" * 60)

    metrics = orchestrator.get_metrics()
    print(f"\nRequests submitted: {metrics['requests_submitted']}")
    print(f"Auto-approved: {metrics.get('requests_auto_approved', 0)}")
    print(f"Approved: {metrics.get('requests_approved', 0)}")
    print(f"Pending: {metrics.get('requests_pending_approval', 0)}")

    print("\nAgent processing counts:")
    for agent, stats in metrics['agents'].items():
        print(f"  {agent}: {stats['processed']} requests")


if __name__ == "__main__":
    asyncio.run(main())
