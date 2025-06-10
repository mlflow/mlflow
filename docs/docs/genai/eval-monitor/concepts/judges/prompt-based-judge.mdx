---
description: 'Learn about prompt-based judges in MLflow - create custom LLM judges with full control over prompts and scoring criteria'
last_update:
  date: 2025-05-18
---

# Prompt-based judges

## Overview

Prompt-based judges enable multi-level quality assessment with customizable choice categories (e.g., excellent/good/poor) and optional numeric scoring. Unlike [guidelines-based judges](/genai/eval-monitor/concepts/judges/guidelines) that provide binary pass/fail evaluation, prompt-based judges offer:

- **Graduated scoring levels** with numeric mapping for tracking improvements
- **Full prompt control** for complex, multi-dimensional evaluation criteria
- **Domain-specific categories** tailored to your use case
- **Aggregatable metrics** to measure quality trends over time

### When to use

**Choose prompt-based judges when you need:**

- Multi-level quality assessment beyond pass/fail
- Numeric scores for quantitative analysis and version comparison
- Complex evaluation criteria requiring custom categories
- Aggregate metrics across datasets

**Choose [guidelines-based judges](/genai/eval-monitor/concepts/judges/guidelines) when you need:**

- Simple pass/fail compliance evaluation
- Business stakeholders to write/update criteria without coding
- Quick iteration on evaluation rules

:::important
While prompt-based judges can be used as a standalone API/SDK, they must be wrapped in a [Scorer](/genai/eval-monitor/concepts/scorers) for use by the [Evaluation Harness](/genai/eval-monitor/concepts/eval-harness) and [production monitoring service](/genai/eval-monitor/concepts/production-monitoring).
:::

## Prerequisites for running the examples

1. Install MLflow and required packages

   ```bash
   pip install --upgrade "mlflow[databricks]>=3.1.0"
   ```

2. Create an MLflow experiment by following the [setup your environment quickstart](/mlflow3/genai/getting-started/connect-environment).

## <a id="quick-start"></a>Example

Here's a simple example that demonstrates the power of prompt-based judges:

```python
from mlflow.genai.judges import create_prompt_judge

# Create a multi-level quality judge
response_quality_judge = create_prompt_judge(
    assessment_name="response_quality",
    prompt_template="""Evaluate the quality of this customer service response:

<request>{{request}}</request>
<response>{{response}}</response>

Choose the most appropriate rating:

[[excellent]]: Empathetic, complete solution, proactive help offered
[[good]]: Addresses the issue adequately with professional tone
[[poor]]: Incomplete, unprofessional, or misses key concerns""",
    numeric_values={
        "excellent": 1.0,
        "good": 0.7,
        "poor": 0.0
    }
)

# Direct usage
feedback = response_quality_judge(
    request="My order arrived damaged!",
    response="I'm so sorry to hear that. I've initiated a replacement order that will arrive tomorrow, and issued a full refund. Is there anything else I can help with?"
)

print(feedback.value)         # 1.0
print(feedback.metadata)      # {"string_value": "excellent"}
print(feedback.rationale)     # Detailed explanation of the rating
```

## <a id="core-concepts"></a>Core concepts

### <a id="sdk-overview"></a>SDK overview

The `create_prompt_judge` function creates a custom LLM judge that evaluates inputs based on your prompt template:

```python
from mlflow.genai.judges import create_prompt_judge

judge = create_prompt_judge(
    assessment_name="formality",
    prompt_template="...",  # Your custom prompt with {{variables}} and [[choices]]
    numeric_values={"formal": 1.0, "informal": 0.0}  # Optional numeric mapping
)

# Returns an mlflow.entities.Feedback object
feedback = judge(request="Hello", response="Hey there!")
```

### <a id="parameters"></a>Parameters

| Parameter         | Type                       | Required | Description                                                                                                                                                                                |
| ----------------- | -------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `assessment_name` | `str`                      | Yes      | Name for the assessment, displayed in MLflow UI and used to identify the judge's output                                                                                                    |
| `prompt_template` | `str`                      | Yes      | Template string containing:<br/>- `{{variables}}`: Placeholders for dynamic content<br/>- `[[choices]]`: Required choice definitions that the judge must select from                       |
| `numeric_values`  | `dict[str, float] \| None` | No       | Maps choice names to numeric scores (0-1 scale recommended).<br/>- Without: Returns string choice values<br/>- With: Returns numeric scores and stores string choice in metadata<br/><br/> |
| `model`           | `str \| None`              | No       | Specific judge model to use (defaults to MLflow's optimized judge model)                                                                                                                   |

### <a id="numeric-mapping"></a>Why use numeric mapping?

When you have multiple choice labels (e.g., "excellent", "good", "poor"), string values make it difficult to track quality improvements across evaluation runs.

Numeric mapping enables:

- **Quantitative comparison**: See if average quality improved from 0.6 to 0.8
- **Aggregate metrics**: Calculate mean scores across your dataset
- **Version comparison**: Track whether changes improved or degraded quality
- **Threshold-based monitoring**: Set alerts when quality drops below acceptable levels

Without numeric values, you can only see label distributions (e.g., 40% "good", 60% "poor"), making it harder to measure overall improvement.

### <a id="return-values"></a>Return values

The function returns a callable that:

- Accepts keyword arguments matching the `{{variables}}` in your prompt template
- Returns an `mlflow.entities.Feedback` object containing:
  - `value`: The selected choice (string) or numeric score if `numeric_values` provided
  - `rationale`: LLM's explanation for its choice
  - `metadata`: Additional information including the string choice when using numeric values
  - `assessment_name`: The name you provided
  - `error`: Error details if evaluation failed

## <a id="prompt-template-requirements"></a>Prompt template requirements

### <a id="choice-format"></a>Choice definition format

Choices must be defined using double square brackets `[[choice_name]]`:

```python
prompt_template = """Evaluate the response formality:

<request>{{request}}</request>
<response>{{response}}</response>

Select one category:

[[formal]]: Professional language, proper grammar, no contractions
[[semi_formal]]: Mix of professional and conversational elements
[[informal]]: Casual language, contractions, colloquialisms"""
```

### <a id="variable-placeholders"></a>Variable placeholders

Use double curly braces `{{variable}}` for dynamic content:

```python
prompt_template = """Assess if the response uses appropriate sources:

Question: {{question}}
Response: {{response}}
Available Sources: {{retrieved_documents}}
Citation Policy: {{citation_policy}}

Choose one:

[[well_cited]]: All claims properly cite available sources
[[partially_cited]]: Some claims cite sources, others do not
[[poorly_cited]]: Claims lack proper citations"""
```

## <a id="common-patterns"></a>Common evaluation patterns

### <a id="likert-scale"></a>Likert scale pattern

Create standard 5-point or 7-point satisfaction scales:

```python
satisfaction_judge = create_prompt_judge(
    assessment_name="customer_satisfaction",
    prompt_template="""Based on this interaction, rate the likely customer satisfaction:

Customer Request: {{request}}
Agent Response: {{response}}

Select satisfaction level:

[[very_satisfied]]: Response exceeds expectations with exceptional service
[[satisfied]]: Response meets expectations adequately
[[neutral]]: Response is acceptable but unremarkable
[[dissatisfied]]: Response fails to meet basic expectations
[[very_dissatisfied]]: Response is unhelpful or problematic""",
    numeric_values={
        "very_satisfied": 1.0,
        "satisfied": 0.75,
        "neutral": 0.5,
        "dissatisfied": 0.25,
        "very_dissatisfied": 0.0
    }
)
```

### <a id="rubric-pattern"></a>Rubric-based scoring

Implement detailed scoring rubrics with clear criteria:

```python
code_review_rubric = create_prompt_judge(
    assessment_name="code_review_rubric",
    prompt_template="""Evaluate this code review using our quality rubric:

Original Code: {{original_code}}
Review Comments: {{review_comments}}
Code Type: {{code_type}}

Score the review quality:

[[comprehensive]]: Identifies all issues including edge cases, security concerns, performance implications, and suggests specific improvements with examples
[[thorough]]: Catches major issues and most minor ones, provides good suggestions but may miss some edge cases
[[adequate]]: Identifies obvious issues and provides basic feedback, misses subtle problems
[[superficial]]: Only catches surface-level issues, feedback is vague or generic
[[inadequate]]: Misses critical issues or provides incorrect feedback""",
    numeric_values={
        "comprehensive": 1.0,
        "thorough": 0.8,
        "adequate": 0.6,
        "superficial": 0.3,
        "inadequate": 0.0
    }
)
```

## <a id="real-world-examples"></a>Real-world examples

### <a id="customer-service"></a>Customer service quality

```python
from mlflow.genai.judges import create_prompt_judge
from mlflow.genai.scorers import scorer
import mlflow

# Issue resolution status judge
resolution_judge = create_prompt_judge(
    assessment_name="issue_resolution",
    prompt_template="""Evaluate if the customer's issue was resolved:

Customer Message: {{customer_message}}
Agent Response: {{agent_response}}
Issue Type: {{issue_type}}

Rate the resolution status:

[[fully_resolved]]: Issue completely addressed with clear solution provided
[[partially_resolved]]: Some progress made but follow-up needed
[[unresolved]]: Issue not addressed or solution unclear
[[escalated]]: Appropriately escalated to higher support tier""",
    numeric_values={
        "fully_resolved": 1.0,
        "partially_resolved": 0.5,
        "unresolved": 0.0,
        "escalated": 0.7  # Positive score for appropriate escalation
    }
)

# Empathy and tone judge
empathy_judge = create_prompt_judge(
    assessment_name="empathy_score",
    prompt_template="""Assess the emotional intelligence of the response:

Customer Emotion: {{customer_emotion}}
Agent Response: {{agent_response}}

Rate the empathy shown:

[[exceptional]]: Acknowledges emotions, validates concerns, shows genuine care
[[good]]: Shows understanding and appropriate concern
[[adequate]]: Professional but somewhat impersonal
[[poor]]: Cold, dismissive, or inappropriate emotional response""",
    numeric_values={
        "exceptional": 1.0,
        "good": 0.75,
        "adequate": 0.5,
        "poor": 0.0
    }
)

# Create a comprehensive customer service scorer
@scorer
def customer_service_quality(inputs, outputs, trace):
    """Comprehensive customer service evaluation"""
    feedbacks = []

    # Evaluate resolution status
    feedbacks.append(resolution_judge(
        customer_message=inputs.get("message", ""),
        agent_response=outputs.get("response", ""),
        issue_type=inputs.get("issue_type", "general")
    ))

    # Evaluate empathy if customer shows emotion
    customer_emotion = inputs.get("detected_emotion", "neutral")
    if customer_emotion in ["frustrated", "angry", "upset", "worried"]:
        feedbacks.append(empathy_judge(
            customer_emotion=customer_emotion,
            agent_response=outputs.get("response", "")
        ))

    return feedbacks

# Example evaluation
eval_data = [
    {
        "inputs": {
            "message": "I've been waiting 3 weeks for my refund! This is unacceptable!",
            "issue_type": "refund",
            "detected_emotion": "angry"
        },
        "outputs": {
            "response": "I completely understand your frustration - 3 weeks is far too long to wait for a refund. I'm escalating this to our finance team immediately. You'll receive your refund within 24 hours, plus a $50 credit for the inconvenience. I'm also sending you my direct email so you can reach me if there are any other delays."
        }
    }
]

results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[customer_service_quality]
)
```

### <a id="content-quality"></a>Content quality assessment

```python
# Technical documentation quality judge
doc_quality_judge = create_prompt_judge(
    assessment_name="documentation_quality",
    prompt_template="""Evaluate this technical documentation:

Content: {{content}}
Target Audience: {{audience}}
Documentation Type: {{doc_type}}

Rate the documentation quality:

[[excellent]]: Clear, complete, well-structured with examples, appropriate depth
[[good]]: Covers topic well, mostly clear, could use minor improvements
[[fair]]: Basic coverage, some unclear sections, missing important details
[[poor]]: Confusing, incomplete, or significantly flawed""",
    numeric_values={
        "excellent": 1.0,
        "good": 0.75,
        "fair": 0.4,
        "poor": 0.0
    }
)

# Marketing copy effectiveness
marketing_judge = create_prompt_judge(
    assessment_name="marketing_effectiveness",
    prompt_template="""Rate this marketing copy's effectiveness:

Copy: {{copy}}
Product: {{product}}
Target Demographic: {{target_demographic}}
Call to Action: {{cta}}

Evaluate effectiveness:

[[highly_effective]]: Compelling, clear value prop, strong CTA, perfect for audience
[[effective]]: Good messaging, decent CTA, reasonably targeted
[[moderately_effective]]: Some good elements but lacks impact or clarity
[[ineffective]]: Weak messaging, unclear value, poor audience fit""",
    numeric_values={
        "highly_effective": 1.0,
        "effective": 0.7,
        "moderately_effective": 0.4,
        "ineffective": 0.0
    }
)
```

### <a id="code-review"></a>Code review quality

```python
# Security review judge
security_review_judge = create_prompt_judge(
    assessment_name="security_review_quality",
    prompt_template="""Evaluate the security aspects of this code review:

Original Code: {{code}}
Review Comments: {{review_comments}}
Security Vulnerabilities Found: {{vulnerabilities_mentioned}}

Rate the security review quality:

[[comprehensive]]: Identifies all security issues, explains risks, suggests secure alternatives
[[thorough]]: Catches major security flaws, good explanations
[[basic]]: Identifies obvious security issues only
[[insufficient]]: Misses critical security vulnerabilities""",
    numeric_values={
        "comprehensive": 1.0,
        "thorough": 0.75,
        "basic": 0.4,
        "insufficient": 0.0
    }
)

# Code clarity feedback judge
code_clarity_judge = create_prompt_judge(
    assessment_name="code_clarity_feedback",
    prompt_template="""Assess the code review's feedback on readability:

Original Code Complexity: {{complexity_score}}
Review Feedback: {{review_comments}}
Readability Issues Identified: {{readability_issues}}

Rate the clarity feedback:

[[excellent]]: Identifies all clarity issues, suggests specific improvements, considers maintainability
[[good]]: Points out main clarity problems with helpful suggestions
[[adequate]]: Basic feedback on obvious readability issues
[[minimal]]: Superficial or missing important clarity feedback""",
    numeric_values={
        "excellent": 1.0,
        "good": 0.7,
        "adequate": 0.4,
        "minimal": 0.0
    }
)
```

### <a id="healthcare"></a>Healthcare communication

```python
# Patient communication appropriateness
patient_comm_judge = create_prompt_judge(
    assessment_name="patient_communication",
    prompt_template="""Evaluate this healthcare provider's response to a patient:

Patient Question: {{patient_question}}
Provider Response: {{provider_response}}
Patient Health Literacy Level: {{health_literacy}}
Sensitive Topics: {{sensitive_topics}}

Rate communication appropriateness:

[[excellent]]: Clear, compassionate, appropriate language level, addresses concerns fully
[[good]]: Generally clear and caring, minor room for improvement
[[acceptable]]: Adequate but could be clearer or more empathetic
[[poor]]: Unclear, uses too much jargon, or lacks appropriate empathy""",
    numeric_values={
        "excellent": 1.0,
        "good": 0.75,
        "acceptable": 0.5,
        "poor": 0.0
    }
)

# Clinical note quality
clinical_note_judge = create_prompt_judge(
    assessment_name="clinical_note_quality",
    prompt_template="""Assess this clinical note's quality:

Note Content: {{note_content}}
Note Type: {{note_type}}
Required Elements: {{required_elements}}

Rate the clinical documentation:

[[comprehensive]]: All required elements present, clear, follows standards, actionable
[[complete]]: Most elements present, generally clear, minor gaps
[[incomplete]]: Missing important elements or lacks clarity
[[deficient]]: Significant gaps, unclear, or doesn't meet documentation standards""",
    numeric_values={
        "comprehensive": 1.0,
        "complete": 0.7,
        "incomplete": 0.3,
        "deficient": 0.0
    }
)
```

### <a id="pairwise-comparison"></a>Pairwise response comparison

Use prompt-based judges to compare two responses and determine which is better. This is useful for A/B testing, model comparison, or preference learning.

:::note
Pairwise comparison judges cannot be used with `mlflow.evaluate()` or as scorers since they evaluate two responses simultaneously rather than a single response. Use them directly for comparative analysis.
:::

```python
from mlflow.genai.judges import create_prompt_judge

# Response preference judge
preference_judge = create_prompt_judge(
    assessment_name="response_preference",
    prompt_template="""Compare these two responses to the same question and determine which is better:

Question: {{question}}

Response A: {{response_a}}

Response B: {{response_b}}

Evaluation Criteria:
1. Accuracy and completeness of information
2. Clarity and ease of understanding
3. Helpfulness and actionability
4. Appropriate tone for the context

Choose your preference:

[[strongly_prefer_a]]: Response A is significantly better across most criteria
[[slightly_prefer_a]]: Response A is marginally better overall
[[equal]]: Both responses are equally good (or equally poor)
[[slightly_prefer_b]]: Response B is marginally better overall
[[strongly_prefer_b]]: Response B is significantly better across most criteria""",
    numeric_values={
        "strongly_prefer_a": -1.0,
        "slightly_prefer_a": -0.5,
        "equal": 0.0,
        "slightly_prefer_b": 0.5,
        "strongly_prefer_b": 1.0
    }
)

# Example usage for model comparison
question = "How do I improve my GenAI app's response quality?"

response_model_v1 = """To improve response quality, you should:
1. Add more training data
2. Fine-tune your model
3. Use better prompts"""

response_model_v2 = """To improve your GenAI app's response quality, consider these strategies:

1. **Enhance your prompts**: Use clear, specific instructions with examples
2. **Implement evaluation**: Use MLflow's LLM judges to measure quality systematically
3. **Collect feedback**: Gather user feedback to identify improvement areas
4. **Iterate on weak areas**: Focus on responses that score poorly
5. **A/B test changes**: Compare versions to ensure improvements

Start with evaluation to establish a baseline, then iterate based on data."""

# Compare responses
feedback = preference_judge(
    question=question,
    response_a=response_model_v1,
    response_b=response_model_v2
)

print(f"Preference: {feedback.metadata['string_value']}")  # "strongly_prefer_b"
print(f"Score: {feedback.value}")  # 1.0
print(f"Rationale: {feedback.rationale}")
```

#### Specialized comparison judges

```python
# Technical accuracy comparison for documentation
tech_comparison_judge = create_prompt_judge(
    assessment_name="technical_comparison",
    prompt_template="""Compare these two technical explanations:

Topic: {{topic}}
Target Audience: {{audience}}

Explanation A: {{explanation_a}}

Explanation B: {{explanation_b}}

Focus on:
- Technical accuracy and precision
- Appropriate depth for the audience
- Use of examples and analogies
- Completeness without overwhelming detail

Which explanation is better?

[[a_much_better]]: A is significantly more accurate and appropriate
[[a_slightly_better]]: A is marginally better in accuracy or clarity
[[equivalent]]: Both are equally good technically
[[b_slightly_better]]: B is marginally better in accuracy or clarity
[[b_much_better]]: B is significantly more accurate and appropriate""",
    numeric_values={
        "a_much_better": -1.0,
        "a_slightly_better": -0.5,
        "equivalent": 0.0,
        "b_slightly_better": 0.5,
        "b_much_better": 1.0
    }
)

# Empathy comparison for customer service
empathy_comparison_judge = create_prompt_judge(
    assessment_name="empathy_comparison",
    prompt_template="""Compare the emotional intelligence of these customer service responses:

Customer Situation: {{situation}}
Customer Emotion: {{emotion}}

Agent Response A: {{response_a}}

Agent Response B: {{response_b}}

Evaluate which response better:
- Acknowledges the customer's emotions
- Shows genuine understanding and care
- Offers appropriate emotional support
- Maintains professional boundaries

Which response shows better emotional intelligence?

[[a_far_superior]]: A shows much better emotional intelligence
[[a_better]]: A is somewhat more empathetic
[[both_good]]: Both show good emotional intelligence
[[b_better]]: B is somewhat more empathetic
[[b_far_superior]]: B shows much better emotional intelligence""",
    numeric_values={
        "a_far_superior": -1.0,
        "a_better": -0.5,
        "both_good": 0.0,
        "b_better": 0.5,
        "b_far_superior": 1.0
    }
)
```

#### Practical comparison workflow

```python
# Compare outputs from different prompt versions
def compare_prompt_versions(test_cases, prompt_v1, prompt_v2, model_client):
    """Compare two prompt versions across multiple test cases"""
    results = []

    for test_case in test_cases:
        # Generate responses with each prompt
        response_v1 = model_client.generate(prompt_v1.format(**test_case))
        response_v2 = model_client.generate(prompt_v2.format(**test_case))

        # Compare responses
        feedback = preference_judge(
            question=test_case["question"],
            response_a=response_v1,
            response_b=response_v2
        )

        results.append({
            "question": test_case["question"],
            "preference": feedback.metadata["string_value"],
            "score": feedback.value,
            "rationale": feedback.rationale
        })

    # Analyze results
    avg_score = sum(r["score"] for r in results) / len(results)

    if avg_score < -0.2:
        print(f"Prompt V1 is preferred (avg score: {avg_score:.2f})")
    elif avg_score > 0.2:
        print(f"Prompt V2 is preferred (avg score: {avg_score:.2f})")
    else:
        print(f"Prompts perform similarly (avg score: {avg_score:.2f})")

    return results

# Compare different model outputs
def compare_models(questions, model_a, model_b, comparison_judge):
    """Compare two models across a set of questions"""
    win_counts = {"model_a": 0, "model_b": 0, "tie": 0}

    for question in questions:
        response_a = model_a.generate(question)
        response_b = model_b.generate(question)

        feedback = comparison_judge(
            question=question,
            response_a=response_a,
            response_b=response_b
        )

        # Count wins based on preference strength
        if feedback.value <= -0.5:
            win_counts["model_a"] += 1
        elif feedback.value >= 0.5:
            win_counts["model_b"] += 1
        else:
            win_counts["tie"] += 1

    print(f"Model comparison results: {win_counts}")
    return win_counts
```

## <a id="advanced-usage"></a>Advanced usage patterns

### <a id="conditional-scoring"></a>Conditional scoring

Implement different evaluation criteria based on context:

```python
@scorer
def adaptive_quality_scorer(inputs, outputs, trace):
    """Applies different judges based on context"""

    # Determine which judge to use based on input characteristics
    query_type = inputs.get("query_type", "general")

    if query_type == "technical":
        judge = create_prompt_judge(
            assessment_name="technical_response",
            prompt_template="""Rate this technical response:

Question: {{question}}
Response: {{response}}
Required Depth: {{depth_level}}

[[expert]]: Demonstrates deep expertise, includes advanced concepts
[[proficient]]: Good technical accuracy, appropriate depth
[[basic]]: Correct but lacks depth or nuance
[[incorrect]]: Contains technical errors or misconceptions""",
            numeric_values={
                "expert": 1.0,
                "proficient": 0.75,
                "basic": 0.5,
                "incorrect": 0.0
            }
        )

        return judge(
            question=inputs["question"],
            response=outputs["response"],
            depth_level=inputs.get("required_depth", "intermediate")
        )

    elif query_type == "support":
        judge = create_prompt_judge(
            assessment_name="support_response",
            prompt_template="""Rate this support response:

Issue: {{issue}}
Response: {{response}}
Customer Status: {{customer_status}}

[[excellent]]: Solves issue completely, proactive, appropriate for customer status
[[good]]: Addresses issue well, professional
[[fair]]: Partially helpful but incomplete
[[poor]]: Unhelpful or inappropriate""",
            numeric_values={
                "excellent": 1.0,
                "good": 0.7,
                "fair": 0.4,
                "poor": 0.0
            }
        )

        return judge(
            issue=inputs["question"],
            response=outputs["response"],
            customer_status=inputs.get("customer_status", "standard")
        )
```

### <a id="score-aggregation"></a>Score aggregation strategies

Combine multiple judge scores intelligently:

```python
@scorer
def weighted_quality_scorer(inputs, outputs, trace):
    """Combines multiple judges with weighted scoring"""

    # Define judges and their weights
    judges_config = [
        {
            "judge": create_prompt_judge(
                assessment_name="accuracy",
                prompt_template="...",  # Your accuracy template
                numeric_values={"high": 1.0, "medium": 0.5, "low": 0.0}
            ),
            "weight": 0.4,
            "args": {"question": inputs["question"], "response": outputs["response"]}
        },
        {
            "judge": create_prompt_judge(
                assessment_name="completeness",
                prompt_template="...",  # Your completeness template
                numeric_values={"complete": 1.0, "partial": 0.5, "incomplete": 0.0}
            ),
            "weight": 0.3,
            "args": {"response": outputs["response"], "requirements": inputs.get("requirements", [])}
        },
        {
            "judge": create_prompt_judge(
                assessment_name="clarity",
                prompt_template="...",  # Your clarity template
                numeric_values={"clear": 1.0, "adequate": 0.6, "unclear": 0.0}
            ),
            "weight": 0.3,
            "args": {"response": outputs["response"]}
        }
    ]

    # Collect all feedbacks
    feedbacks = []
    weighted_score = 0.0

    for config in judges_config:
        feedback = config["judge"](**config["args"])
        feedbacks.append(feedback)

        # Add to weighted score if numeric
        if isinstance(feedback.value, (int, float)):
            weighted_score += feedback.value * config["weight"]

    # Add composite score as additional feedback
    from mlflow.entities import Feedback
    composite_feedback = Feedback(
        name="weighted_quality_score",
        value=weighted_score,
        rationale=f"Weighted combination of {len(judges_config)} quality dimensions"
    )
    feedbacks.append(composite_feedback)

    return feedbacks
```

## <a id="best-practices"></a>Best practices

### <a id="choice-design"></a>Designing effective choices

**1. Make choices mutually exclusive and exhaustive**

```python
# Good - clear distinctions, covers all cases
"""[[approved]]: Meets all requirements, ready for production
[[needs_revision]]: Has issues that must be fixed before approval
[[rejected]]: Fundamental flaws, requires complete rework"""

# Bad - overlapping and ambiguous
"""[[good]]: The response is good
[[okay]]: The response is okay
[[fine]]: The response is fine"""
```

**2. Provide specific criteria for each choice**

```python
# Good - specific, measurable criteria
"""[[secure]]: No vulnerabilities, follows all security best practices, includes input validation
[[mostly_secure]]: Minor security concerns that should be addressed but aren't critical
[[insecure]]: Contains vulnerabilities that could be exploited"""

# Bad - vague criteria
"""[[secure]]: Looks secure
[[not_secure]]: Has problems"""
```

**3. Order choices logically (best to worst)**

```python
# Good - clear progression
numeric_values = {
    "exceptional": 1.0,
    "good": 0.75,
    "satisfactory": 0.5,
    "needs_improvement": 0.25,
    "unacceptable": 0.0
}
```

### <a id="numeric-scale-design"></a>Numeric scale design

**1. Use consistent scales across judges**

```python
# All judges use 0-1 scale
quality_judge = create_prompt_judge(..., numeric_values={"high": 1.0, "medium": 0.5, "low": 0.0})
accuracy_judge = create_prompt_judge(..., numeric_values={"accurate": 1.0, "partial": 0.5, "wrong": 0.0})
```

**2. Leave gaps for future refinement**

```python
# Allows adding intermediate levels later
numeric_values = {
    "excellent": 1.0,
    "good": 0.7,    # Gap allows for "very_good" at 0.85
    "fair": 0.4,    # Gap allows for "satisfactory" at 0.55
    "poor": 0.0
}
```

**3. Consider domain-specific scales**

```python
# Academic grading scale
academic_scale = {
    "A": 4.0,
    "B": 3.0,
    "C": 2.0,
    "D": 1.0,
    "F": 0.0
}

# Net Promoter Score scale
nps_scale = {
    "promoter": 1.0,      # 9-10
    "passive": 0.0,       # 7-8
    "detractor": -1.0     # 0-6
}
```

### <a id="prompt-engineering"></a>Prompt engineering tips

**1. Structure prompts clearly**

```python
prompt_template = """[Clear Task Description]
Evaluate the technical accuracy of this response.

[Context Section]
Question: {{question}}
Response: {{response}}
Technical Domain: {{domain}}

[Evaluation Criteria]
Consider: factual accuracy, appropriate depth, correct terminology

[Choice Definitions]
[[accurate]]: All technical facts correct, appropriate level of detail
[[mostly_accurate]]: Minor inaccuracies that don't affect core understanding
[[inaccurate]]: Contains significant errors or misconceptions"""
```

**2. Include examples when helpful**

```python
prompt_template = """Assess the urgency level of this support ticket.

Ticket: {{ticket_content}}

Examples of each level:
- Critical: System down, data loss, security breach
- High: Major feature broken, blocking work
- Medium: Performance issues, non-critical bugs
- Low: Feature requests, minor UI issues

Choose urgency level:
[[critical]]: Immediate attention required, business impact
[[high]]: Urgent, significant user impact
[[medium]]: Important but not urgent
[[low]]: Can be addressed in normal workflow"""
```

## <a id="comparison-with-guidelines"></a>Comparison with guidelines-based judges

| Aspect                     | Guidelines-based                       | Prompt-based                                                   |
| -------------------------- | -------------------------------------- | -------------------------------------------------------------- |
| **Evaluation type**        | Binary pass/fail                       | Multi-level categories                                         |
| **Scoring**                | "yes" or "no"                          | Custom choices with optional numeric values                    |
| **Best for**               | Compliance, policy adherence           | Quality assessment, satisfaction ratings                       |
| **Iteration speed**        | Very fast - just update guideline text | Moderate - may need to adjust choices                          |
| **Business user friendly** | ✅ High - natural language rules       | ⚠️ Medium - requires understanding choices and the full prompt |
| **Aggregation**            | Count pass/fail rates                  | Calculate averages, track trends                               |

## <a id="validation-error-handling"></a>Validation and error handling

### <a id="choice-validation"></a>Choice validation

The judge validates that:

- Choices are properly defined with `[[choice_name]]` format
- Choice names are alphanumeric (can include underscores)
- At least one choice is defined in the template

```python
# This will raise an error - no choices defined
invalid_judge = create_prompt_judge(
    assessment_name="invalid",
    prompt_template="Rate the response: {{response}}"
)
# ValueError: Prompt template must include choices denoted with [[CHOICE_NAME]]
```

### <a id="numeric-validation"></a>Numeric values validation

When using `numeric_values`, all choices must be mapped:

```python
# This will raise an error - missing choice in numeric_values
invalid_judge = create_prompt_judge(
    assessment_name="invalid",
    prompt_template="""Choose:
    [[option_a]]: First option
    [[option_b]]: Second option""",
    numeric_values={"option_a": 1.0}  # Missing option_b
)
# ValueError: numeric_values keys must match the choices
```

### <a id="template-validation"></a>Template variable validation

Missing template variables raise errors during execution:

```python
judge = create_prompt_judge(
    assessment_name="test",
    prompt_template="{{request}} {{response}} [[good]]: Good"
)

# This will raise an error - missing 'response' variable
judge(request="Hello")
# KeyError: Template variable 'response' not found
```

## Next Steps

- [Create prompt-based scorers](/genai/eval-monitor/custom-judge/create-prompt-judge) - Step-by-step guide to implement prompt-based judges
- [Guidelines-based judges](/genai/eval-monitor/concepts/judges/guidelines) - Simpler alternative for pass/fail criteria
- [Custom scorers overview](/genai/eval-monitor/custom-scorers) - Learn how to wrap judges in custom scorers
