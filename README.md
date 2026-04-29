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
├── ch01-identity/           # Enterprise identity and authentication
├── ch02-gateway/            # API keys and secrets management  
├── ch03-registry/           # Agent registry and service discovery
├── ch04-anomaly/            # Anomaly detection and monitoring
├── ch05-evaluation/         # Agent evaluation frameworks
├── ch06-observability/      # Observability and instrumentation
├── ch07-debugging/          # Debugging tools and techniques
├── ch08-cost/               # Cost tracking and attribution
├── ch09-infrastructure/     # Infrastructure and scaling
├── ch10-human-agent/        # Human-agent collaboration patterns
├── ch13-customer-service/   # Case study: Customer service platform
├── ch14-procurement/        # Case study: Enterprise procurement
├── common/                  # Shared utilities and type definitions
└── requirements.txt         # Python dependencies
```

## Chapter Contents

| Chapter | Topic | Key Implementations |
|---------|-------|---------------------|
| 1 | Enterprise Identity | JWT auth, X.509 certificates, OIDC delegation, credential lifecycle |
| 2 | API Gateway | Policy enforcement, request validation, secrets management |
| 3 | Agent Registry | Service discovery, health monitoring, capability matching |
| 4 | Anomaly Detection | Statistical anomaly detection, behavioral drift, cost anomalies |
| 5 | Evaluation | Graders, evaluation datasets, quality metrics |
| 6 | Observability | OpenTelemetry instrumentation, distributed tracing, metrics |
| 7 | Debugging | Breakpoints, step-through execution, state inspection |
| 8 | Cost Tracking | Token attribution, budget management, chargeback reporting |
| 9 | Infrastructure | Queue-based scaling, worker pools, auto-scaling |
| 10 | Human-Agent | HITL workflows, approval gates, feedback collection |
| 13 | Customer Service | Multi-agent support platform with escalation workflows |
| 14 | Procurement | Enterprise procurement with approval chains |

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

# Run the customer support platform
python ch13-customer-service/support_agents.py

# Run the procurement system
python ch14-procurement/procurement_agents.py
```

## Key Topics

### Part I: Governance (Chapters 1-4)
Enterprise-grade identity, authentication, secrets management, rate limiting, and audit logging for agent systems.

### Part II: Operations (Chapters 5-8)
Deployment strategies, monitoring and observability, error handling, and testing frameworks.

### Part III: Infrastructure (Chapters 9-10)
Scaling patterns, state management, and distributed systems considerations.

### Part IV: Case Studies (Chapters 13-14)
Complete, production-ready examples demonstrating all concepts in realistic enterprise scenarios.

## Technology Stack

- **Authentication**: PyJWT, cryptography
- **Observability**: OpenTelemetry, Prometheus
- **Caching**: Redis, aiocache
- **Infrastructure**: Docker, Kubernetes
- **Testing**: pytest, pytest-asyncio

## Code Quality

This codebase follows modern Python best practices:

- **Security**: JWT secrets loaded from environment variables, safe AST-based evaluation instead of `eval()`, proper exception handling with logging
- **Python 3.12+ Compatibility**: Uses `datetime.now(timezone.utc)` instead of deprecated `datetime.utcnow()`
- **Redis 7.0 Compatibility**: Uses `BLMOVE` instead of deprecated `BRPOPLPUSH`
- **Thread Safety**: Rate limiters and circuit breakers use proper locking for concurrent access
- **Import Standards**: PEP 8 compliant import ordering throughout

### Validation

All code has been validated through a comprehensive gauntlet process:

```
✅ Syntax check: All 15 Python files pass
✅ Security scan: No hardcoded secrets or unsafe patterns
✅ Deprecation check: No deprecated Python APIs
✅ Style check: Consistent formatting and imports
```

## Related Resources

- **Book**: *Agents in Production: Operating Multi-Agent Systems at Scale*
- **Companion Volume**: [Agent Architectures](https://github.com/vijaygwu/Agent-Architectures) - Design Patterns for Multi-Agent AI Systems

## License

This code is provided for educational purposes to accompany the book. See LICENSE for details.

## Author

**Vijay Raghavan**

- GitHub: [@vijaygwu](https://github.com/vijaygwu)
