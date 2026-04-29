"""
Chapter 13: Case Study - Enterprise Customer Service Platform
=============================================================

A production-ready customer support system demonstrating:
- Intent classification and routing
- Multi-agent specialization
- Escalation workflows
- Knowledge base integration
- Conversation context management
- Quality monitoring

Handles customer inquiries from intake through resolution.
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

class TicketPriority(Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class TicketCategory(Enum):
    """Categories of support tickets."""
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    PRODUCT = "product"
    RETURNS = "returns"
    GENERAL = "general"


class TicketStatus(Enum):
    """Status of a support ticket."""
    NEW = "new"
    TRIAGING = "triaging"
    IN_PROGRESS = "in_progress"
    WAITING_CUSTOMER = "waiting_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class SentimentLevel(Enum):
    """Customer sentiment levels."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


@dataclass
class Message:
    """A message in a support conversation."""
    id: str
    content: str
    sender: str  # "customer" or agent_id
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class Customer:
    """Customer information."""
    id: str
    name: str
    email: str
    tier: str = "standard"  # standard, premium, enterprise
    account_age_days: int = 0
    previous_tickets: int = 0


@dataclass
class SupportTicket:
    """A customer support ticket."""
    id: str
    customer: Customer
    subject: str
    messages: list[Message] = field(default_factory=list)
    category: Optional[TicketCategory] = None
    priority: TicketPriority = TicketPriority.MEDIUM
    status: TicketStatus = TicketStatus.NEW
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    assigned_agent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    resolution_summary: str = ""
    tags: list[str] = field(default_factory=list)
    escalation_history: list[dict] = field(default_factory=list)

    def add_message(self, content: str, sender: str, metadata: Optional[dict] = None):
        """Add a message to the ticket."""
        msg = Message(
            id=f"msg_{len(self.messages)}",
            content=content,
            sender=sender,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.updated_at = time.time()
        return msg


@dataclass
class KnowledgeArticle:
    """A knowledge base article."""
    id: str
    title: str
    content: str
    category: TicketCategory
    tags: list[str] = field(default_factory=list)
    views: int = 0
    helpfulness_score: float = 0.0


# =============================================================================
# Agent Definitions
# =============================================================================

class SupportAgent:
    """Base class for support agents."""

    def __init__(self, agent_id: str, name: str, specialization: Optional[TicketCategory] = None):
        self.agent_id = agent_id
        self.name = name
        self.specialization = specialization
        self.handled_count = 0
        self.resolved_count = 0
        self.avg_resolution_time = 0.0

    async def process(self, ticket: SupportTicket, context: dict) -> dict:
        """Process a ticket. Override in subclasses."""
        raise NotImplementedError


class TriageAgent(SupportAgent):
    """
    Classifies and routes incoming tickets.
    """

    def __init__(self):
        super().__init__("triage-agent", "Triage Agent")
        self.category_keywords = {
            TicketCategory.BILLING: ["bill", "charge", "payment", "invoice", "refund", "price"],
            TicketCategory.TECHNICAL: ["error", "bug", "crash", "not working", "slow", "issue"],
            TicketCategory.ACCOUNT: ["password", "login", "account", "profile", "settings"],
            TicketCategory.PRODUCT: ["feature", "how to", "tutorial", "documentation"],
            TicketCategory.RETURNS: ["return", "exchange", "warranty", "damaged"],
        }
        self.priority_signals = {
            "urgent": ["urgent", "emergency", "asap", "immediately", "critical"],
            "angry": ["terrible", "awful", "worst", "unacceptable", "lawsuit"]
        }

    async def process(self, ticket: SupportTicket, context: dict) -> dict:
        """Classify and prioritize the ticket."""
        self.handled_count += 1

        # Get all message content
        all_text = " ".join([m.content.lower() for m in ticket.messages])
        all_text += " " + ticket.subject.lower()

        # Classify category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for kw in keywords if kw in all_text)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            ticket.category = max(category_scores.keys(), key=lambda k: category_scores[k])
        else:
            ticket.category = TicketCategory.GENERAL

        # Assess priority
        priority = TicketPriority.MEDIUM

        # Check for urgency signals
        if any(word in all_text for word in self.priority_signals["urgent"]):
            priority = TicketPriority.HIGH

        if any(word in all_text for word in self.priority_signals["angry"]):
            priority = TicketPriority.HIGH
            ticket.sentiment = SentimentLevel.ANGRY

        # Premium customers get higher priority
        if ticket.customer.tier in ["premium", "enterprise"]:
            if priority == TicketPriority.MEDIUM:
                priority = TicketPriority.HIGH
            elif priority == TicketPriority.HIGH:
                priority = TicketPriority.URGENT

        ticket.priority = priority
        ticket.status = TicketStatus.TRIAGING

        # Generate tags
        ticket.tags = self._extract_tags(all_text)

        return {
            "category": ticket.category.value,
            "priority": ticket.priority.value,
            "sentiment": ticket.sentiment.value,
            "tags": ticket.tags,
            "confidence": 0.85 if category_scores else 0.5
        }

    def _extract_tags(self, text: str) -> list[str]:
        """Extract relevant tags from text."""
        tags = []

        tag_keywords = {
            "mobile": ["mobile", "app", "ios", "android", "phone"],
            "api": ["api", "endpoint", "integration", "webhook"],
            "performance": ["slow", "lag", "timeout", "performance"],
            "security": ["security", "breach", "hack", "unauthorized"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in text for kw in keywords):
                tags.append(tag)

        return tags


class BillingAgent(SupportAgent):
    """
    Handles billing-related inquiries.
    """

    def __init__(self):
        super().__init__("billing-agent", "Billing Specialist", TicketCategory.BILLING)
        self.billing_actions = ["refund", "credit", "upgrade", "downgrade", "cancel"]

    async def process(self, ticket: SupportTicket, context: dict) -> dict:
        """Process billing inquiry."""
        self.handled_count += 1

        last_message = ticket.messages[-1].content.lower() if ticket.messages else ""

        # Identify billing action needed
        action = None
        for a in self.billing_actions:
            if a in last_message:
                action = a
                break

        # Prepare response
        response = await self._generate_response(ticket, action, context)

        # Check if needs human escalation
        needs_escalation = False
        escalation_reason = None

        if action == "refund" and self._estimate_refund_amount(context) > 100:
            needs_escalation = True
            escalation_reason = "High-value refund requires supervisor approval"

        if action == "cancel" and ticket.customer.tier == "enterprise":
            needs_escalation = True
            escalation_reason = "Enterprise cancellation requires account manager"

        # Add response to ticket
        ticket.add_message(response["message"], self.agent_id)
        ticket.status = TicketStatus.IN_PROGRESS

        return {
            "action_identified": action,
            "response": response["message"],
            "needs_escalation": needs_escalation,
            "escalation_reason": escalation_reason,
            "suggested_resolution": response.get("resolution")
        }

    async def _generate_response(self, ticket: SupportTicket,
                                   action: Optional[str], context: dict) -> dict:
        """Generate response for billing inquiry."""
        customer_name = ticket.customer.name

        responses = {
            "refund": {
                "message": f"Hi {customer_name}, I understand you're requesting a refund. "
                          "I'm reviewing your account now. Could you please confirm which "
                          "transaction you'd like refunded?",
                "resolution": "Process refund after confirmation"
            },
            "credit": {
                "message": f"Hi {customer_name}, I'd be happy to help with account credits. "
                          "Let me check your account history to see what options are available.",
                "resolution": "Apply appropriate credit"
            },
            "upgrade": {
                "message": f"Hi {customer_name}, great choice! Let me walk you through our "
                          "upgrade options and find the best plan for your needs.",
                "resolution": "Process plan upgrade"
            },
            "default": {
                "message": f"Hi {customer_name}, thank you for reaching out about your billing. "
                          "I'm here to help. Could you provide more details about your concern?",
                "resolution": "Gather more information"
            }
        }

        return responses.get(action, responses["default"])

    def _estimate_refund_amount(self, context: dict) -> float:
        """Estimate refund amount from context."""
        return context.get("estimated_refund", 0)


class TechnicalAgent(SupportAgent):
    """
    Handles technical support issues.
    """

    def __init__(self, knowledge_base: list[KnowledgeArticle]):
        super().__init__("tech-agent", "Technical Specialist", TicketCategory.TECHNICAL)
        self.knowledge_base = knowledge_base
        self.diagnostic_questions = [
            "What error message are you seeing?",
            "When did this issue start?",
            "Have you tried restarting the application?",
            "What browser/device are you using?"
        ]

    async def process(self, ticket: SupportTicket, context: dict) -> dict:
        """Process technical issue."""
        self.handled_count += 1

        # Search knowledge base
        relevant_articles = await self._search_knowledge_base(ticket)

        # Analyze issue complexity
        complexity = await self._assess_complexity(ticket)

        # Generate response
        if relevant_articles:
            response = await self._generate_kb_response(ticket, relevant_articles[0])
        else:
            response = await self._generate_diagnostic_response(ticket)

        ticket.add_message(response["message"], self.agent_id)
        ticket.status = TicketStatus.IN_PROGRESS

        # Check if needs escalation
        needs_escalation = complexity == "high" or "security" in ticket.tags

        return {
            "complexity": complexity,
            "relevant_articles": [a.id for a in relevant_articles],
            "response": response["message"],
            "needs_escalation": needs_escalation,
            "escalation_reason": "Complex technical issue" if needs_escalation else None,
            "next_steps": response.get("next_steps", [])
        }

    async def _search_knowledge_base(self, ticket: SupportTicket) -> list[KnowledgeArticle]:
        """Search knowledge base for relevant articles."""
        all_text = " ".join([m.content.lower() for m in ticket.messages])
        all_text += " " + ticket.subject.lower()

        scored_articles = []
        for article in self.knowledge_base:
            score = 0
            for tag in article.tags:
                if tag in all_text:
                    score += 2
            if article.category == ticket.category:
                score += 1
            if score > 0:
                scored_articles.append((score, article))

        scored_articles.sort(key=lambda x: x[0], reverse=True)
        return [a for _, a in scored_articles[:3]]

    async def _assess_complexity(self, ticket: SupportTicket) -> str:
        """Assess issue complexity."""
        # Simple heuristic based on message count and tags
        if len(ticket.messages) > 5:
            return "high"
        if "api" in ticket.tags or "security" in ticket.tags:
            return "high"
        if "performance" in ticket.tags:
            return "medium"
        return "low"

    async def _generate_kb_response(self, ticket: SupportTicket,
                                     article: KnowledgeArticle) -> dict:
        """Generate response using knowledge base article."""
        return {
            "message": f"Hi {ticket.customer.name}, thank you for reporting this issue. "
                      f"Based on your description, this article may help: '{article.title}'. "
                      f"\n\n{article.content[:200]}... \n\n"
                      "Please let me know if this resolves your issue.",
            "next_steps": ["Try suggested solution", "Follow up if not resolved"]
        }

    async def _generate_diagnostic_response(self, ticket: SupportTicket) -> dict:
        """Generate diagnostic questions."""
        asked_questions = set()
        for msg in ticket.messages:
            if msg.sender != "customer":
                asked_questions.add(msg.content)

        # Find next question to ask
        for q in self.diagnostic_questions:
            if q not in asked_questions:
                return {
                    "message": f"Hi {ticket.customer.name}, I'm looking into this. "
                              f"To help troubleshoot, {q}",
                    "next_steps": ["Await customer response", "Continue diagnostics"]
                }

        return {
            "message": f"Hi {ticket.customer.name}, thank you for the details. "
                      "I'm investigating this further and will get back to you shortly.",
            "next_steps": ["Escalate to engineering if needed"]
        }


class EscalationAgent(SupportAgent):
    """
    Handles escalated tickets requiring human intervention.
    """

    def __init__(self):
        super().__init__("escalation-agent", "Escalation Manager")
        self.human_queue: list[SupportTicket] = []
        self.escalation_callbacks: list[Callable] = []

    def on_escalation(self, callback: Callable):
        """Register callback for escalation events."""
        self.escalation_callbacks.append(callback)

    async def process(self, ticket: SupportTicket, context: dict) -> dict:
        """Process escalation."""
        self.handled_count += 1

        reason = context.get("escalation_reason", "Complex issue")

        ticket.escalation_history.append({
            "timestamp": time.time(),
            "from_agent": context.get("from_agent"),
            "reason": reason,
            "priority": ticket.priority.value
        })

        ticket.status = TicketStatus.ESCALATED

        # Queue for human
        self.human_queue.append(ticket)

        # Notify callbacks
        for callback in self.escalation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(ticket, reason)
                else:
                    callback(ticket, reason)
            except Exception as e:
                logging.warning(f"Escalation callback failed: {e}")

        # Send acknowledgment to customer
        ack_message = (
            f"Hi {ticket.customer.name}, your case has been escalated to our "
            "senior support team for further attention. A specialist will "
            "reach out to you shortly. Thank you for your patience."
        )
        ticket.add_message(ack_message, self.agent_id)

        return {
            "status": "escalated",
            "queue_position": len(self.human_queue),
            "reason": reason,
            "estimated_response": "Within 2 hours" if ticket.priority in [
                TicketPriority.HIGH, TicketPriority.URGENT
            ] else "Within 24 hours"
        }


# =============================================================================
# Orchestrator
# =============================================================================

class CustomerSupportOrchestrator:
    """
    Orchestrates the multi-agent customer support workflow.
    """

    def __init__(self, knowledge_base: list[KnowledgeArticle] = None):
        self.triage = TriageAgent()
        self.billing = BillingAgent()
        self.technical = TechnicalAgent(knowledge_base or [])
        self.escalation = EscalationAgent()

        self.tickets: dict[str, SupportTicket] = {}
        self.metrics = defaultdict(int)
        self._event_handlers: dict[str, list[Callable]] = defaultdict(list)

        # Agent routing
        self.category_agents = {
            TicketCategory.BILLING: self.billing,
            TicketCategory.TECHNICAL: self.technical,
        }

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
            except Exception as e:
                logging.warning(f"Event handler for '{event}' failed: {e}")

    async def create_ticket(self, customer: Customer, subject: str,
                             initial_message: str) -> SupportTicket:
        """Create a new support ticket."""
        ticket = SupportTicket(
            id=f"TKT-{int(time.time())}-{len(self.tickets)}",
            customer=customer,
            subject=subject
        )
        ticket.add_message(initial_message, "customer")

        self.tickets[ticket.id] = ticket
        self.metrics["tickets_created"] += 1

        await self._emit("ticket_created", ticket)

        return ticket

    async def process_ticket(self, ticket_id: str) -> dict:
        """Process a ticket through the support workflow."""
        if ticket_id not in self.tickets:
            return {"error": "Ticket not found"}

        ticket = self.tickets[ticket_id]

        # Triage
        triage_result = await self.triage.process(ticket, {})
        await self._emit("ticket_triaged", ticket)

        # Route to specialized agent
        agent = self.category_agents.get(ticket.category)
        if not agent:
            # Default handling
            agent = self.technical

        context = {"from_triage": triage_result}
        result = await agent.process(ticket, context)

        # Check for escalation
        if result.get("needs_escalation"):
            escalation_result = await self.escalation.process(ticket, {
                "escalation_reason": result.get("escalation_reason"),
                "from_agent": agent.agent_id
            })
            self.metrics["tickets_escalated"] += 1
            return {
                "ticket_id": ticket.id,
                "status": "escalated",
                "triage": triage_result,
                "handling": result,
                "escalation": escalation_result
            }

        return {
            "ticket_id": ticket.id,
            "status": "in_progress",
            "triage": triage_result,
            "handling": result
        }

    async def add_customer_response(self, ticket_id: str, message: str) -> dict:
        """Add a customer response to a ticket."""
        if ticket_id not in self.tickets:
            return {"error": "Ticket not found"}

        ticket = self.tickets[ticket_id]
        ticket.add_message(message, "customer")

        if ticket.status == TicketStatus.WAITING_CUSTOMER:
            ticket.status = TicketStatus.IN_PROGRESS

        # Re-process with new message
        return await self.process_ticket(ticket_id)

    async def resolve_ticket(self, ticket_id: str, resolution: str) -> dict:
        """Resolve a ticket."""
        if ticket_id not in self.tickets:
            return {"error": "Ticket not found"}

        ticket = self.tickets[ticket_id]
        ticket.status = TicketStatus.RESOLVED
        ticket.resolved_at = time.time()
        ticket.resolution_summary = resolution

        # Calculate resolution time
        resolution_time = ticket.resolved_at - ticket.created_at

        # Update agent metrics
        if ticket.assigned_agent:
            for agent in [self.triage, self.billing, self.technical, self.escalation]:
                if agent.agent_id == ticket.assigned_agent:
                    agent.resolved_count += 1
                    # Update average resolution time
                    prev_avg = agent.avg_resolution_time
                    n = agent.resolved_count
                    agent.avg_resolution_time = (prev_avg * (n-1) + resolution_time) / n

        self.metrics["tickets_resolved"] += 1
        await self._emit("ticket_resolved", ticket)

        return {
            "ticket_id": ticket.id,
            "status": "resolved",
            "resolution_time_seconds": resolution_time,
            "resolution": resolution
        }

    def get_queue_status(self) -> dict:
        """Get current queue status."""
        by_status = defaultdict(int)
        by_category = defaultdict(int)
        by_priority = defaultdict(int)

        for ticket in self.tickets.values():
            by_status[ticket.status.value] += 1
            if ticket.category:
                by_category[ticket.category.value] += 1
            by_priority[ticket.priority.value] += 1

        return {
            "total": len(self.tickets),
            "by_status": dict(by_status),
            "by_category": dict(by_category),
            "by_priority": dict(by_priority),
            "escalation_queue": len(self.escalation.human_queue)
        }

    def get_metrics(self) -> dict:
        """Get system metrics."""
        return {
            **dict(self.metrics),
            "queue": self.get_queue_status(),
            "agents": {
                agent.name: {
                    "handled": agent.handled_count,
                    "resolved": agent.resolved_count,
                    "avg_resolution_time": agent.avg_resolution_time
                }
                for agent in [self.triage, self.billing, self.technical, self.escalation]
            }
        }


# =============================================================================
# Example Usage
# =============================================================================

async def main():
    """Demonstration of customer support system."""
    print("=" * 60)
    print("Multi-Agent Customer Support System")
    print("=" * 60)

    # Initialize knowledge base
    kb = [
        KnowledgeArticle(
            "kb-001", "How to reset your password",
            "To reset your password, go to Settings > Security > Reset Password...",
            TicketCategory.ACCOUNT, ["password", "login"]
        ),
        KnowledgeArticle(
            "kb-002", "Troubleshooting slow performance",
            "If you're experiencing slow performance, try: 1. Clear cache 2. Restart...",
            TicketCategory.TECHNICAL, ["slow", "performance", "cache"]
        ),
        KnowledgeArticle(
            "kb-003", "Understanding your bill",
            "Your monthly bill includes base subscription plus usage charges...",
            TicketCategory.BILLING, ["bill", "invoice", "charge"]
        )
    ]

    # Initialize system
    orchestrator = CustomerSupportOrchestrator(knowledge_base=kb)

    # Event handlers
    async def on_escalation(ticket, reason):
        print(f"\n  [ESCALATION] Ticket {ticket.id}: {reason}")

    orchestrator.escalation.on_escalation(on_escalation)

    print("\n" + "-" * 40)
    print("Scenario 1: Simple billing inquiry")
    print("-" * 40)

    customer1 = Customer("cust-001", "John Smith", "john@example.com", "standard", 180, 2)
    ticket1 = await orchestrator.create_ticket(
        customer1,
        "Question about my bill",
        "Hi, I noticed an extra charge on my bill this month. Can you explain?"
    )

    result1 = await orchestrator.process_ticket(ticket1.id)
    print(f"\nTicket {ticket1.id}:")
    print(f"  Category: {result1['triage']['category']}")
    print(f"  Priority: {result1['triage']['priority']}")
    print(f"  Status: {result1['status']}")

    print("\n" + "-" * 40)
    print("Scenario 2: Technical issue with KB match")
    print("-" * 40)

    customer2 = Customer("cust-002", "Alice Wong", "alice@example.com", "premium", 365, 5)
    ticket2 = await orchestrator.create_ticket(
        customer2,
        "Application is very slow",
        "The app has been really slow for the past few days. Pages take forever to load."
    )

    result2 = await orchestrator.process_ticket(ticket2.id)
    print(f"\nTicket {ticket2.id}:")
    print(f"  Category: {result2['triage']['category']}")
    print(f"  Priority: {result2['triage']['priority']}")
    print(f"  Status: {result2['status']}")
    if result2['handling'].get('relevant_articles'):
        print(f"  KB Articles: {result2['handling']['relevant_articles']}")

    print("\n" + "-" * 40)
    print("Scenario 3: Angry enterprise customer (escalation)")
    print("-" * 40)

    customer3 = Customer("cust-003", "Bob Enterprise", "bob@enterprise.com", "enterprise", 730, 15)
    ticket3 = await orchestrator.create_ticket(
        customer3,
        "URGENT: Critical billing error",
        "This is unacceptable! We've been overcharged $5000 and I need this fixed immediately. "
        "This is the worst service I've experienced. I want a full refund."
    )

    result3 = await orchestrator.process_ticket(ticket3.id)
    print(f"\nTicket {ticket3.id}:")
    print(f"  Category: {result3['triage']['category']}")
    print(f"  Priority: {result3['triage']['priority']}")
    print(f"  Sentiment: {result3['triage']['sentiment']}")
    print(f"  Status: {result3['status']}")
    if result3.get('escalation'):
        print(f"  Queue Position: {result3['escalation']['queue_position']}")

    # Resolve first ticket
    print("\n" + "-" * 40)
    print("Resolving Ticket 1")
    print("-" * 40)

    resolve_result = await orchestrator.resolve_ticket(
        ticket1.id,
        "Explained the charge was for additional storage. Customer satisfied."
    )
    print(f"  Resolution time: {resolve_result['resolution_time_seconds']:.1f}s")

    # Show metrics
    print("\n" + "=" * 60)
    print("System Metrics")
    print("=" * 60)

    metrics = orchestrator.get_metrics()
    print(f"\nTickets created: {metrics['tickets_created']}")
    print(f"Tickets resolved: {metrics.get('tickets_resolved', 0)}")
    print(f"Tickets escalated: {metrics.get('tickets_escalated', 0)}")

    print("\nQueue Status:")
    queue = metrics['queue']
    print(f"  Total: {queue['total']}")
    print(f"  By status: {queue['by_status']}")
    print(f"  By priority: {queue['by_priority']}")

    print("\nAgent Performance:")
    for name, stats in metrics['agents'].items():
        print(f"  {name}: handled={stats['handled']}, resolved={stats['resolved']}")


if __name__ == "__main__":
    asyncio.run(main())
