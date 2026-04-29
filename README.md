# Agents in Production

**Companion Code Repository**

*Operating Multi-Agent Systems at Scale*

by Vijay Raghavan

---

## About This Repository

This repository contains the complete Python implementations for all code examples in *Agents in Production: Operating Multi-Agent Systems at Scale*. Each chapter's code is organized in its own directory with production-ready implementations covering governance, operations, and infrastructure.

## Book Overview

*Agents in Production* is your operations manual for running multi-agent AI systems at scale. While the companion volume covers design patterns, this book focuses on everything that happens after deployment: identity and authentication, secrets management, observability, scaling, and real-world case studies.

## Repository Structure

```
.
├── ch01-identity/           # Identity service with JWT and certificate auth
├── ch02-gateway/            # API gateway with rate limiting and routing
├── ch03-registry/           # Agent registry and discovery service
├── ch04-anomaly/            # Anomaly detection for agent behavior
├── ch05-evaluation/         # Evaluation framework for agent performance
├── ch06-observability/      # Observability with metrics and tracing
├── ch07-debugging/          # Debugging tools and log analysis
├── ch08-cost/               # Cost attribution and budget management
├── ch09-infrastructure/     # Infrastructure scaling and state management
├── ch10-human-agent/        # Human-agent collaboration patterns
├── ch13-procurement/        # Enterprise procurement system example
├── ch14-support/            # Customer support platform example
├── common/                  # Shared utilities and type definitions
└── requirements.txt         # Python dependencies
```

## Chapter Contents

| Chapter | Topic | Key Implementations |
|---------|-------|---------------------|
| 1 | Enterprise Identity | JWT auth, certificate management, OIDC integration |
| 2 | API Gateway | Rate limiting, request routing, load balancing |
| 3 | Agent Registry | Service discovery, health checks, capability matching |
| 4 | Anomaly Detection | Behavioral analysis, drift detection, alerting |
| 5 | Evaluation | Performance metrics, A/B testing, quality assessment |
| 6 | Observability | OpenTelemetry tracing, Prometheus metrics, dashboards |
| 7 | Debugging | Log aggregation, trace analysis, root cause identification |
| 8 | Cost Attribution | Token tracking, budget allocation, cost optimization |
| 9 | Infrastructure | Auto-scaling, state management, queue-based architecture |
| 10 | Human-Agent Collaboration | Handoff protocols, escalation, feedback loops |
| 13 | Procurement System | Complete enterprise procurement example |
| 14 | Support Platform | Complete customer service example |

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Redis (for caching and state management examples)
- Docker (optional, for infrastructure examples)
- An API key for your preferred LLM provider

### Installation

```bash
# Clone the repository
git clone https://github.com/vijaygwu/Agents-in-Production.git
cd Agents-in-Production

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Set your environment variables:

```bash
# LLM API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Infrastructure (optional)
export REDIS_URL="redis://localhost:6379"
export JAEGER_ENDPOINT="http://localhost:14268/api/traces"
```

### Running Examples

Each chapter directory contains standalone examples:

```bash
# Run the identity service example
python ch01-identity/identity_service.py

# Run the observability example
python ch06-observability/observability.py

# Run the complete procurement system
python ch13-procurement/procurement_agents.py

# Run the customer support platform
python ch14-support/support_agents.py
```

## Key Topics

### Governance (Chapters 1-4)
Enterprise-grade identity, authentication, rate limiting, and behavioral monitoring for agent systems.

### Operations (Chapters 5-8)
Evaluation frameworks, observability pipelines, debugging tools, and cost management.

### Infrastructure (Chapters 9-10)
Scaling patterns, state management, and human-agent collaboration protocols.

### Case Studies (Chapters 13-14)
Complete, production-ready examples demonstrating all concepts in realistic enterprise scenarios.

## Technology Stack

- **Authentication**: PyJWT, cryptography
- **Observability**: OpenTelemetry, Prometheus
- **Caching**: Redis, aiocache
- **Infrastructure**: Docker, Kubernetes
- **Testing**: pytest, pytest-asyncio

## Related Resources

- **Book**: *Agents in Production: Operating Multi-Agent Systems at Scale*
- **Companion Volume**: [Agent Architectures](https://github.com/vijaygwu/Agent-Architectures) - Design Patterns for Multi-Agent AI Systems

## License

This code is provided for educational purposes to accompany the book. See LICENSE for details.

## Author

**Vijay Raghavan**

- GitHub: [@vijaygwu](https://github.com/vijaygwu)
