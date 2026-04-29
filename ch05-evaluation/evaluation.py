"""
Chapter 5: Agent Deployment Strategies
======================================

Implements a comprehensive evaluation framework for AI agents:
- Code-based graders
- Model-based graders (LLM-as-judge)
- Human grading integration
- Evaluation datasets and test cases
- Metrics aggregation and reporting

Based on Google's agent evaluation approaches and industry best practices.
"""

import asyncio
import json
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import defaultdict


class GradeLevel(Enum):
    """Grading scale for evaluations."""
    EXCELLENT = 5
    GOOD = 4
    ACCEPTABLE = 3
    POOR = 2
    FAIL = 1


@dataclass
class TestCase:
    """A single test case for agent evaluation."""
    id: str
    input: dict
    expected_output: Optional[dict] = None
    expected_behavior: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"
    metadata: dict = field(default_factory=dict)


@dataclass
class GradeResult:
    """Result of grading an agent response."""
    score: float  # 0.0 to 1.0
    grade: GradeLevel
    grader_id: str
    explanation: str
    details: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_score(cls, score: float, grader_id: str, explanation: str, **kwargs) -> 'GradeResult':
        """Create GradeResult from numeric score."""
        if score >= 0.9:
            grade = GradeLevel.EXCELLENT
        elif score >= 0.75:
            grade = GradeLevel.GOOD
        elif score >= 0.5:
            grade = GradeLevel.ACCEPTABLE
        elif score >= 0.25:
            grade = GradeLevel.POOR
        else:
            grade = GradeLevel.FAIL

        return cls(score=score, grade=grade, grader_id=grader_id, explanation=explanation, **kwargs)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a test case."""
    test_case_id: str
    agent_id: str
    input: dict
    output: Any
    grades: list[GradeResult]
    aggregate_score: float = 0.0
    passed: bool = False
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def calculate_aggregate(self, weights: Optional[dict[str, float]] = None):
        """Calculate aggregate score from individual grades."""
        if not self.grades:
            self.aggregate_score = 0.0
            self.passed = False
            return

        weights = weights or {}
        total_weight = 0.0
        weighted_sum = 0.0

        for grade in self.grades:
            weight = weights.get(grade.grader_id, 1.0)
            weighted_sum += grade.score * weight
            total_weight += weight

        self.aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        self.passed = self.aggregate_score >= 0.5


class Grader(ABC):
    """Base class for all graders."""

    def __init__(self, grader_id: str):
        self.grader_id = grader_id

    @abstractmethod
    async def grade(self,
                    test_case: TestCase,
                    output: Any,
                    context: dict) -> GradeResult:
        """Grade the agent output for a test case."""
        pass


class ExactMatchGrader(Grader):
    """
    Code-based grader that checks for exact matches.
    """

    def __init__(self, grader_id: str = "exact_match",
                 case_sensitive: bool = True,
                 normalize_whitespace: bool = True):
        super().__init__(grader_id)
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace

    async def grade(self, test_case: TestCase, output: Any, context: dict) -> GradeResult:
        start = time.time()

        if test_case.expected_output is None:
            return GradeResult.from_score(
                0.0, self.grader_id,
                "No expected output defined for exact match",
                latency_ms=(time.time() - start) * 1000
            )

        expected = self._normalize(str(test_case.expected_output))
        actual = self._normalize(str(output))

        if expected == actual:
            return GradeResult.from_score(
                1.0, self.grader_id,
                "Exact match",
                latency_ms=(time.time() - start) * 1000
            )

        return GradeResult.from_score(
            0.0, self.grader_id,
            f"No match. Expected: '{expected[:100]}...', Got: '{actual[:100]}...'",
            latency_ms=(time.time() - start) * 1000
        )

    def _normalize(self, text: str) -> str:
        if not self.case_sensitive:
            text = text.lower()
        if self.normalize_whitespace:
            text = ' '.join(text.split())
        return text


class ContainsGrader(Grader):
    """
    Code-based grader that checks if output contains expected elements.
    """

    def __init__(self, grader_id: str = "contains",
                 required_elements: Optional[list[str]] = None,
                 forbidden_elements: Optional[list[str]] = None):
        super().__init__(grader_id)
        self.required_elements = required_elements or []
        self.forbidden_elements = forbidden_elements or []

    async def grade(self, test_case: TestCase, output: Any, context: dict) -> GradeResult:
        start = time.time()
        output_str = str(output).lower()

        missing = []
        for element in self.required_elements:
            if element.lower() not in output_str:
                missing.append(element)

        forbidden_found = []
        for element in self.forbidden_elements:
            if element.lower() in output_str:
                forbidden_found.append(element)

        total_checks = len(self.required_elements) + len(self.forbidden_elements)
        if total_checks == 0:
            return GradeResult.from_score(
                1.0, self.grader_id,
                "No elements to check",
                latency_ms=(time.time() - start) * 1000
            )

        passed_checks = (
            len(self.required_elements) - len(missing) +
            len(self.forbidden_elements) - len(forbidden_found)
        )
        score = passed_checks / total_checks

        explanation_parts = []
        if missing:
            explanation_parts.append(f"Missing: {missing}")
        if forbidden_found:
            explanation_parts.append(f"Forbidden found: {forbidden_found}")
        if not explanation_parts:
            explanation_parts.append("All checks passed")

        return GradeResult.from_score(
            score, self.grader_id,
            "; ".join(explanation_parts),
            details={"missing": missing, "forbidden_found": forbidden_found},
            latency_ms=(time.time() - start) * 1000
        )


class JSONSchemaGrader(Grader):
    """
    Code-based grader that validates JSON output against a schema.
    """

    def __init__(self, grader_id: str = "json_schema",
                 required_fields: Optional[list[str]] = None,
                 field_types: Optional[dict[str, type]] = None):
        super().__init__(grader_id)
        self.required_fields = required_fields or []
        self.field_types = field_types or {}

    async def grade(self, test_case: TestCase, output: Any, context: dict) -> GradeResult:
        start = time.time()

        # Parse JSON if string
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError as e:
                return GradeResult.from_score(
                    0.0, self.grader_id,
                    f"Invalid JSON: {e}",
                    latency_ms=(time.time() - start) * 1000
                )

        if not isinstance(output, dict):
            return GradeResult.from_score(
                0.0, self.grader_id,
                f"Expected dict, got {type(output).__name__}",
                latency_ms=(time.time() - start) * 1000
            )

        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")

        # Check field types
        for field, expected_type in self.field_types.items():
            if field in output:
                if not isinstance(output[field], expected_type):
                    errors.append(
                        f"Field '{field}' has wrong type: expected {expected_type.__name__}, "
                        f"got {type(output[field]).__name__}"
                    )

        total_checks = len(self.required_fields) + len(self.field_types)
        if total_checks == 0:
            return GradeResult.from_score(
                1.0, self.grader_id,
                "No schema checks defined",
                latency_ms=(time.time() - start) * 1000
            )

        score = 1.0 - (len(errors) / total_checks)

        return GradeResult.from_score(
            score, self.grader_id,
            f"{len(errors)} errors" if errors else "Schema validation passed",
            details={"errors": errors},
            latency_ms=(time.time() - start) * 1000
        )


class FunctionGrader(Grader):
    """
    Code-based grader using a custom function.
    """

    def __init__(self, grader_id: str,
                 grade_func: Callable[[TestCase, Any, dict], tuple[float, str]]):
        super().__init__(grader_id)
        self.grade_func = grade_func

    async def grade(self, test_case: TestCase, output: Any, context: dict) -> GradeResult:
        start = time.time()

        try:
            if asyncio.iscoroutinefunction(self.grade_func):
                score, explanation = await self.grade_func(test_case, output, context)
            else:
                score, explanation = self.grade_func(test_case, output, context)

            return GradeResult.from_score(
                score, self.grader_id,
                explanation,
                latency_ms=(time.time() - start) * 1000
            )
        except Exception as e:
            return GradeResult.from_score(
                0.0, self.grader_id,
                f"Grading function error: {e}",
                latency_ms=(time.time() - start) * 1000
            )


class LLMGrader(Grader):
    """
    Model-based grader using an LLM as judge.
    Implements the "LLM-as-judge" evaluation pattern.
    """

    def __init__(self, grader_id: str = "llm_judge",
                 model_call: Optional[Callable] = None,
                 criteria: Optional[list[str]] = None):
        super().__init__(grader_id)
        self.model_call = model_call
        self.criteria = criteria or [
            "accuracy",
            "completeness",
            "relevance",
            "clarity"
        ]

    async def grade(self, test_case: TestCase, output: Any, context: dict) -> GradeResult:
        start = time.time()

        # Build evaluation prompt
        prompt = self._build_evaluation_prompt(test_case, output)

        if self.model_call:
            try:
                response = await self._call_model(prompt)
                score, explanation = self._parse_response(response)
            except Exception as e:
                return GradeResult.from_score(
                    0.0, self.grader_id,
                    f"LLM grading error: {e}",
                    latency_ms=(time.time() - start) * 1000
                )
        else:
            # Simulate for demo
            score, explanation = self._simulated_grade(test_case, output)

        return GradeResult.from_score(
            score, self.grader_id,
            explanation,
            details={"criteria": self.criteria},
            latency_ms=(time.time() - start) * 1000
        )

    def _build_evaluation_prompt(self, test_case: TestCase, output: Any) -> str:
        criteria_text = "\n".join(f"- {c}" for c in self.criteria)
        return f"""Evaluate the following agent response:

INPUT:
{json.dumps(test_case.input, indent=2)}

EXPECTED BEHAVIOR:
{test_case.expected_behavior or 'Not specified'}

ACTUAL OUTPUT:
{json.dumps(output, indent=2) if isinstance(output, (dict, list)) else str(output)}

Evaluate based on these criteria:
{criteria_text}

Provide a score from 0.0 to 1.0 and a brief explanation.
Format: SCORE: <number>
EXPLANATION: <text>"""

    async def _call_model(self, prompt: str) -> str:
        if asyncio.iscoroutinefunction(self.model_call):
            return await self.model_call(prompt)
        return self.model_call(prompt)

    def _parse_response(self, response: str) -> tuple[float, str]:
        lines = response.strip().split('\n')
        score = 0.5
        explanation = response

        for line in lines:
            if line.startswith('SCORE:'):
                try:
                    score = float(line.split(':')[1].strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith('EXPLANATION:'):
                explanation = line.split(':', 1)[1].strip()

        return score, explanation

    def _simulated_grade(self, test_case: TestCase, output: Any) -> tuple[float, str]:
        """Simulated grading for demo purposes."""
        output_str = str(output).lower()

        # Simple heuristic scoring
        score = 0.5  # Base score

        # Check if output mentions expected topics
        if test_case.expected_behavior:
            keywords = test_case.expected_behavior.lower().split()
            matches = sum(1 for kw in keywords if kw in output_str)
            score += 0.3 * (matches / max(len(keywords), 1))

        # Length check (reasonable output length)
        if 50 < len(output_str) < 2000:
            score += 0.1

        # Structure check
        if isinstance(output, dict):
            score += 0.1

        score = min(1.0, score)
        return score, f"Simulated LLM evaluation: {score:.2f}"


class HumanGrader(Grader):
    """
    Human grading interface.
    Queues items for human review and integrates results.
    """

    def __init__(self, grader_id: str = "human"):
        super().__init__(grader_id)
        self.pending_reviews: dict[str, dict] = {}
        self.completed_reviews: dict[str, GradeResult] = {}

    async def grade(self, test_case: TestCase, output: Any, context: dict) -> GradeResult:
        review_id = hashlib.sha256(
            f"{test_case.id}:{time.time()}".encode()
        ).hexdigest()[:12]

        # Queue for human review
        self.pending_reviews[review_id] = {
            "test_case": test_case,
            "output": output,
            "context": context,
            "created_at": time.time()
        }

        # Return pending result
        return GradeResult.from_score(
            0.5, self.grader_id,  # Default score until reviewed
            f"Pending human review (id: {review_id})",
            details={"review_id": review_id, "status": "pending"}
        )

    def submit_review(self, review_id: str, score: float, explanation: str) -> bool:
        """Submit human review result."""
        if review_id not in self.pending_reviews:
            return False

        self.completed_reviews[review_id] = GradeResult.from_score(
            score, self.grader_id,
            f"Human review: {explanation}",
            details={"review_id": review_id, "status": "completed"}
        )

        del self.pending_reviews[review_id]
        return True

    def get_pending_reviews(self) -> list[dict]:
        """Get all pending human reviews."""
        return [
            {"review_id": rid, **review}
            for rid, review in self.pending_reviews.items()
        ]


class EvaluationDataset:
    """
    Collection of test cases for evaluation.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.test_cases: list[TestCase] = []
        self.metadata: dict = {}

    def add_test_case(self, test_case: TestCase):
        """Add a test case to the dataset."""
        self.test_cases.append(test_case)

    def add_test_cases(self, test_cases: list[TestCase]):
        """Add multiple test cases."""
        self.test_cases.extend(test_cases)

    def filter(self,
               tags: Optional[list[str]] = None,
               difficulty: Optional[str] = None,
               category: Optional[str] = None) -> list[TestCase]:
        """Filter test cases by criteria."""
        results = self.test_cases

        if tags:
            results = [tc for tc in results if any(t in tc.tags for t in tags)]

        if difficulty:
            results = [tc for tc in results if tc.difficulty == difficulty]

        if category:
            results = [tc for tc in results if tc.category == category]

        return results

    def sample(self, n: int, stratify_by: Optional[str] = None) -> list[TestCase]:
        """Sample test cases from dataset."""
        import random

        if stratify_by:
            # Group by attribute
            groups = defaultdict(list)
            for tc in self.test_cases:
                key = getattr(tc, stratify_by, "default")
                groups[key].append(tc)

            # Sample proportionally from each group
            results = []
            per_group = max(1, n // len(groups))
            for group_cases in groups.values():
                results.extend(random.sample(
                    group_cases,
                    min(per_group, len(group_cases))
                ))
            return results[:n]

        return random.sample(self.test_cases, min(n, len(self.test_cases)))


class EvaluationFramework:
    """
    Main evaluation framework that orchestrates grading.
    """

    def __init__(self):
        self.graders: list[Grader] = []
        self.grader_weights: dict[str, float] = {}
        self.results: list[EvaluationResult] = []

    def add_grader(self, grader: Grader, weight: float = 1.0):
        """Add a grader to the framework."""
        self.graders.append(grader)
        self.grader_weights[grader.grader_id] = weight

    async def evaluate(self,
                       agent_func: Callable,
                       test_case: TestCase,
                       agent_id: str = "default") -> EvaluationResult:
        """Evaluate agent on a single test case."""
        start_time = time.time()

        # Execute agent
        try:
            if asyncio.iscoroutinefunction(agent_func):
                output = await agent_func(test_case.input)
            else:
                output = agent_func(test_case.input)
            error = None
        except Exception as e:
            output = None
            error = str(e)

        execution_time = (time.time() - start_time) * 1000

        # Grade with all graders
        grades = []
        context = {
            "test_case_id": test_case.id,
            "agent_id": agent_id,
            "execution_time_ms": execution_time
        }

        for grader in self.graders:
            grade = await grader.grade(test_case, output, context)
            grades.append(grade)

        # Create result
        result = EvaluationResult(
            test_case_id=test_case.id,
            agent_id=agent_id,
            input=test_case.input,
            output=output,
            grades=grades,
            execution_time_ms=execution_time,
            error=error
        )

        result.calculate_aggregate(self.grader_weights)
        self.results.append(result)

        return result

    async def evaluate_dataset(self,
                                agent_func: Callable,
                                dataset: EvaluationDataset,
                                agent_id: str = "default",
                                parallel: int = 1) -> list[EvaluationResult]:
        """Evaluate agent on entire dataset."""
        if parallel == 1:
            results = []
            for test_case in dataset.test_cases:
                result = await self.evaluate(agent_func, test_case, agent_id)
                results.append(result)
            return results
        else:
            # Parallel evaluation
            semaphore = asyncio.Semaphore(parallel)

            async def eval_with_semaphore(tc):
                async with semaphore:
                    return await self.evaluate(agent_func, tc, agent_id)

            return await asyncio.gather(
                *[eval_with_semaphore(tc) for tc in dataset.test_cases]
            )

    def get_metrics(self,
                    agent_id: Optional[str] = None,
                    category: Optional[str] = None) -> dict:
        """Calculate evaluation metrics."""
        results = self.results

        if agent_id:
            results = [r for r in results if r.agent_id == agent_id]

        if not results:
            return {"error": "No results found"}

        scores = [r.aggregate_score for r in results]
        pass_count = sum(1 for r in results if r.passed)

        by_grader = defaultdict(list)
        for result in results:
            for grade in result.grades:
                by_grader[grade.grader_id].append(grade.score)

        return {
            "total_cases": len(results),
            "pass_rate": pass_count / len(results),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "by_grader": {
                gid: {
                    "average": sum(scores) / len(scores),
                    "count": len(scores)
                }
                for gid, scores in by_grader.items()
            },
            "error_rate": sum(1 for r in results if r.error) / len(results),
            "avg_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results)
        }

    def generate_report(self, agent_id: Optional[str] = None) -> str:
        """Generate human-readable evaluation report."""
        metrics = self.get_metrics(agent_id)

        if "error" in metrics:
            return f"Error: {metrics['error']}"

        report = []
        report.append("=" * 60)
        report.append("AGENT EVALUATION REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Test Cases: {metrics['total_cases']}")
        report.append(f"Pass Rate: {metrics['pass_rate']:.1%}")
        report.append(f"Average Score: {metrics['average_score']:.3f}")
        report.append(f"Score Range: {metrics['min_score']:.3f} - {metrics['max_score']:.3f}")
        report.append(f"Error Rate: {metrics['error_rate']:.1%}")
        report.append(f"Avg Execution Time: {metrics['avg_execution_time_ms']:.1f}ms")

        report.append("\n" + "-" * 40)
        report.append("Scores by Grader:")
        for grader_id, stats in metrics['by_grader'].items():
            report.append(f"  {grader_id}: {stats['average']:.3f} ({stats['count']} grades)")

        return "\n".join(report)


# =============================================================================
# Example Usage
# =============================================================================

async def example_agent(input_data: dict) -> dict:
    """Example agent for demonstration."""
    await asyncio.sleep(0.01)  # Simulate processing

    query = input_data.get("query", "")

    # Simple response generation
    return {
        "answer": f"Response to: {query}",
        "confidence": 0.85,
        "sources": ["source1", "source2"]
    }


async def main():
    """Demonstration of evaluation framework."""
    print("=" * 60)
    print("Agent Evaluation Framework Demonstration")
    print("=" * 60)

    # Create framework
    framework = EvaluationFramework()

    # Add graders
    framework.add_grader(JSONSchemaGrader(
        required_fields=["answer", "confidence"],
        field_types={"confidence": float}
    ), weight=1.0)

    framework.add_grader(ContainsGrader(
        grader_id="content_check",
        required_elements=["response"],
        forbidden_elements=["error", "failed"]
    ), weight=0.8)

    framework.add_grader(LLMGrader(
        criteria=["relevance", "helpfulness", "accuracy"]
    ), weight=1.5)

    # Add custom function grader
    def custom_grader(test_case: TestCase, output: Any, context: dict) -> tuple[float, str]:
        if isinstance(output, dict) and output.get("confidence", 0) > 0.7:
            return 1.0, "High confidence response"
        return 0.5, "Low confidence response"

    framework.add_grader(FunctionGrader("confidence_check", custom_grader), weight=0.5)

    # Create test dataset
    dataset = EvaluationDataset("qa_tests", "Question answering evaluation")

    dataset.add_test_cases([
        TestCase(
            id="qa-001",
            input={"query": "What is the capital of France?"},
            expected_behavior="Should provide Paris as the answer",
            tags=["geography", "factual"],
            difficulty="easy",
            category="qa"
        ),
        TestCase(
            id="qa-002",
            input={"query": "Explain quantum computing"},
            expected_behavior="Should provide a clear explanation of quantum computing concepts",
            tags=["science", "explanation"],
            difficulty="hard",
            category="qa"
        ),
        TestCase(
            id="qa-003",
            input={"query": "How do I make pasta?"},
            expected_behavior="Should provide cooking instructions",
            tags=["cooking", "how-to"],
            difficulty="easy",
            category="qa"
        ),
        TestCase(
            id="qa-004",
            input={"query": "Compare Python and JavaScript"},
            expected_behavior="Should compare the two programming languages",
            tags=["programming", "comparison"],
            difficulty="medium",
            category="qa"
        ),
        TestCase(
            id="qa-005",
            input={"query": "What are the benefits of exercise?"},
            expected_behavior="Should list health benefits of exercise",
            tags=["health", "factual"],
            difficulty="easy",
            category="qa"
        )
    ])

    print(f"\nDataset: {dataset.name}")
    print(f"Test cases: {len(dataset.test_cases)}")
    print(f"Graders: {[g.grader_id for g in framework.graders]}")

    # Run evaluation
    print("\n" + "-" * 40)
    print("Running evaluation...")

    results = await framework.evaluate_dataset(
        example_agent,
        dataset,
        agent_id="test-agent"
    )

    # Print individual results
    print("\n" + "-" * 40)
    print("Individual Results:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"\n  [{status}] {result.test_case_id}")
        print(f"      Score: {result.aggregate_score:.3f}")
        print(f"      Time: {result.execution_time_ms:.1f}ms")
        for grade in result.grades:
            print(f"      - {grade.grader_id}: {grade.score:.2f} ({grade.explanation[:50]}...)")

    # Generate report
    print("\n" + framework.generate_report())

    # Filter test cases
    print("\n" + "-" * 40)
    print("Filtered Tests:")
    easy_tests = dataset.filter(difficulty="easy")
    print(f"  Easy tests: {len(easy_tests)}")

    factual_tests = dataset.filter(tags=["factual"])
    print(f"  Factual tests: {len(factual_tests)}")


if __name__ == "__main__":
    asyncio.run(main())
