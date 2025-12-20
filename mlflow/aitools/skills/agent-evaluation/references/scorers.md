# MLflow Evaluation Scorers Guide

Complete guide for selecting and creating scorers to evaluate agent quality.

## Table of Contents

1. [Understanding Scorers](#understanding-scorers)
2. [Built-in Scorers](#built-in-scorers)
3. [Custom Scorer Design](#custom-scorer-design)
4. [Pass/Fail Format (Recommended)](#passfail-format-recommended)
5. [Scorer Registration](#scorer-registration)
6. [Testing Scorers](#testing-scorers)

## Understanding Scorers

### What are Scorers?

Scorers (also called "judges" or "LLM-as-a-judge") are evaluation criteria that assess the quality of agent responses. They:
- Take agent inputs and outputs as input
- Apply quality criteria (relevance, accuracy, completeness, etc.)
- Return a score or pass/fail judgment
- Can be built-in (provided by MLflow) or custom (defined by you)

### Types of Scorers

**1. Reference-Free Scorers**
- Don't require ground truth or expected outputs
- Judge quality based on the query and response alone
- Examples: Relevance, Completeness, Clarity
- **Easiest to use** - work with any dataset

**2. Ground-Truth Scorers**
- Require expected outputs in the dataset
- Compare agent response to ground truth
- Examples: Factual Accuracy, Answer Correctness
- Require datasets with `expectations` field

### LLM-as-a-Judge Pattern

Modern scorers use an LLM to judge quality:
1. Scorer receives query and response
2. LLM is given evaluation instructions
3. LLM judges whether criteria is met
4. Returns structured output (pass/fail or numeric)

## Built-in Scorers

MLflow provides several built-in scorers for common evaluation criteria.

### Accessing Built-in Scorers

**Always use the Documentation Access Protocol:**

1. Read `https://mlflow.org/docs/latest/llms.txt`
2. Search for keywords: "judges", "scorers", "LLM-as-a-judge", "evaluation"
3. Follow links to detailed scorer documentation
4. Review available scorers and their requirements

### Checking Available Scorers

List registered scorers in your environment:

```bash
uv run mlflow scorers
```

Output shows:
- Scorer names
- Whether they're built-in or custom
- Registration details

### Common Built-in Scorers

**Note**: Exact list depends on MLflow version. Consult documentation for current list.

**Typical built-in scorers include:**

**RelevanceToQuery**
- Judges if response addresses the question
- Reference-free
- Use for: All agents

**Completeness**
- Judges if answer is complete
- Reference-free
- Use for: Question answering, information retrieval

**Groundedness**
- Judges if response is grounded in provided context
- May require ground truth context
- Use for: RAG systems, knowledge base agents

**Coherence**
- Judges if response is logically coherent
- Reference-free
- Use for: All agents, especially conversational

**Answer Correctness**
- Compares response to expected answer
- Requires ground truth
- Use for: When you have known correct answers

### Important: Trace Structure Assumptions

**CRITICAL**: Built-in scorers make assumptions about trace structure.

Before using a built-in scorer:
1. **Read its documentation** to understand required inputs
2. **Check trace structure** matches expectations
3. **Verify it works** with a test trace before full evaluation

**Example issue**:
- Scorer expects `context` field in trace
- Your agent doesn't provide `context`
- Scorer fails or returns null

**Solution**:
- Read scorer docs carefully
- Test on single trace first
- Create custom scorer if built-in doesn't match your structure

### Using Built-in Scorers

Built-in scorers can be used in two ways:

**Option 1: Direct use without registration** (always available):

```bash
mlflow traces evaluate \
  --trace-ids <trace_id> \
  --scorers RelevanceToQuery,Completeness \
  --output json
```

**Option 2: Register to experiment** (recommended for consistency):

Registering built-in scorers to your experiment makes them show up when listing scorers:

```python
import os
from mlflow.genai.scorers import Correctness, Relevance

# Register built-in scorer to experiment
scorer = Correctness()
scorer.register(experiment_id=os.getenv("MLFLOW_EXPERIMENT_ID"))
```

**Available built-in scorers** (check MLflow docs for complete list):
- `Correctness`
- `Relevance`
- `Faithfulness`
- `Coherence`
- And others...

**Benefits of registration**:
- Shows up in `mlflow scorers list --experiment-id <id>`
- Keeps all evaluation criteria in one place
- Makes it clear what scorers are being used for the experiment

## Custom Scorer Design

Create custom scorers when:
- Built-in scorers don't match your criteria
- You need domain-specific evaluation
- Your agent has unique requirements
- Trace structure doesn't match built-in assumptions

### Design Process

**Step 1: Define Quality Criterion Clearly**

What specific aspect of quality are you judging?

**Examples**:
- "Response uses appropriate tools for the query"
- "Response is factually accurate based on available data"
- "Response follows the expected format"
- "Response appropriately handles ambiguous queries"

**Step 2: Determine Required Inputs**

What information does the scorer need?

**Common inputs**:
- `query`: The user's question
- `response`: The agent's answer
- `trace`: Full trace with tool calls, LLM calls, etc.
- `context`: Retrieved documents or context (if applicable)

**Step 3: Write Evaluation Instructions**

Clear instructions for the LLM judge:

```
You are evaluating whether an agent used appropriate tools for a query.

Given:
- Query: {query}
- Response: {response}
- Trace: {trace} (contains tool calls)

Criteria: The agent should use tools when needed (e.g., search for factual queries)
and should not use tools unnecessarily (e.g., for greetings).

Evaluate whether appropriate tools were used. Return "pass" if tools were used
appropriately, "fail" if not.
```

**Step 4: Choose Output Format**

See [Pass/Fail Format](#passfail-format-recommended) below.

**Strongly recommend: Pass/Fail format**

### Example Custom Scorers

**Example 1: Tool Usage Appropriateness**

```
Scorer Name: ToolUsageAppropriate

Definition: Judges whether the agent used appropriate tools for the query.

Instructions:
You are evaluating tool usage by an AI agent.

Given a query and trace showing tool calls, determine if:
1. Tools were used when needed (factual questions, searches, lookups)
2. Tools were NOT used unnecessarily (greetings, simple questions)
3. The RIGHT tools were chosen for the task

Return "pass" if tool usage was appropriate, "fail" if not.

Variables: query, response, trace
Output: pass/fail
```

**Example 2: Factual Accuracy**

```
Scorer Name: FactualAccuracy

Definition: Judges whether the response is factually accurate.

Instructions:
You are evaluating the factual accuracy of an AI agent's response.

Review the agent's response and determine if the information provided is
factually correct based on the context and your knowledge.

Return "pass" if the response is factually accurate, "fail" if it contains
incorrect information or makes unsupported claims.

Variables: query, response, context (optional)
Output: pass/fail
```

## Pass/Fail Format (Recommended)

### Why Pass/Fail?

**Strongly recommend using pass/fail format** over numeric scales.

**Benefits**:
1. **Easier to interpret**: "Did it pass?" vs "What does 3.5 mean?"
2. **Simpler aggregation**: Pass rate = passed / total
3. **Better for filtering**: "Show me all failures"
4. **Clear actionability**: Pass = good, Fail = needs work
5. **Consistent across scorers**: All scorers use same scale

### Format Specification

**Pass values**: `"pass"`, `"yes"`, `"true"`, `"passed"`
**Fail values**: `"fail"`, `"no"`, `"false"`, `"failed"`

**Prefer**: `"pass"` and `"fail"` for consistency.

### Pass/Fail vs Numeric Scales

**❌ Avoid Numeric Scales**:
```
Output: 1-5 scale
- 1 = Poor
- 2 = Below Average
- 3 = Average
- 4 = Good
- 5 = Excellent

Problems:
- Hard to interpret (is 3 good enough?)
- Subjective (one judge's 3 is another's 4)
- Difficult to aggregate
- Unclear thresholds
```

**✅ Prefer Pass/Fail**:
```
Output: pass or fail
- pass = Meets criteria
- fail = Does not meet criteria

Benefits:
- Clear interpretation
- Objective threshold
- Easy aggregation (% passed)
- Actionable
```

### Example Pass/Fail Instructions

**Good:**
```
Evaluate whether the response is complete.

A response is COMPLETE if it fully answers all parts of the question
without omitting important information.

Return "pass" if the response is complete, "fail" if incomplete.
```

**Bad:**
```
Rate the completeness of the response on a scale of 1-5, where:
1 = Very incomplete
2 = Mostly incomplete
3 = Somewhat complete
4 = Mostly complete
5 = Very complete
```

## Scorer Registration

### Using register-llm-judge CLI

**Recommended method** for most custom scorers:

```bash
uv run mlflow scorers register-llm-judge \
  --name "ToolUsageAppropriate" \
  --definition "Judges whether appropriate tools were used for the query" \
  --instructions "You are evaluating tool usage by an AI agent..." \
  --variables query,response,trace \
  --output pass/fail
```

**Parameters**:
- `--name`: Scorer name (used to reference it in evaluation)
- `--definition`: Short description of what it evaluates
- `--instructions`: Complete evaluation instructions for the LLM judge
- `--variables`: Comma-separated list of inputs (query, response, trace, context)
- `--output`: Output format (prefer "pass/fail")
- `--model` (optional): LLM to use (default is usually fine)

### Using make_judge() Function

**Programmatic registration** for advanced use cases:

```python
from mlflow.llms.evaluation import make_judge

scorer = make_judge(
    name="ToolUsageAppropriate",
    definition="Judges whether appropriate tools were used",
    instructions="""
    You are evaluating tool usage by an AI agent.

    Given a query and trace, determine if appropriate tools were used.
    Return "pass" if tool usage was appropriate, "fail" if not.
    """,
    variables=["query", "response", "trace"],
    output_type="pass/fail",
    # model="gpt-4",  # Optional, defaults to configured model
)

# Register the scorer
mlflow.log_scorer(scorer, name="ToolUsageAppropriate")
```

**When to use make_judge()**:
- Need programmatic control
- Complex scorer logic
- Integration with existing code
- Dynamic scorer generation

### Best Practices

**1. Use default model** unless you have specific needs:
- Default is usually sufficient and cost-effective
- Specify model only for specialized evaluation

**2. Do NOT register built-in scorers**:
- Built-in scorers are already available
- Registration only needed for custom scorers

**3. Test before full evaluation**:
- Test on single trace first
- Verify output format is correct
- Check that instructions are clear

**4. Version your scorers**:
- Include version in name if criteria change: `ToolUsageAppropriate_v2`
- Document what changed between versions

## Testing Scorers

**Always test a scorer** before using it on your full evaluation dataset.

### Quick Single-Trace Test

```bash
# Get a sample trace ID (from previous agent run)
export TRACE_ID="<sample_trace_id>"

# Test scorer on single trace
uv run mlflow traces evaluate \
  --output json \
  --scorers ToolUsageAppropriate \
  --trace-ids $TRACE_ID
```

### Verify Scorer Behavior

**Check 1: No Errors**
- Scorer executes without errors
- No null or empty outputs

**Check 2: Output Format**
- For pass/fail: Returns "pass" or "fail"
- For numeric: Returns number in expected range

**Check 3: Makes Sense**
- Review the trace
- Manually judge if scorer output is reasonable
- If scorer is wrong, refine instructions

**Check 4: Trace Coverage**
- Test on diverse traces (different query types)
- Ensure scorer handles all cases
- Check edge cases

### Iteration Workflow

1. **Register scorer** with initial instructions
2. **Test on single trace** with known expected outcome
3. **Review output** - does it match your judgment?
4. **If wrong**: Refine instructions, re-register, test again
5. **Test on diverse traces** (3-5 different types)
6. **Deploy to full evaluation** once confident

### Example Test Session

```bash
# Test ToolUsageAppropriate scorer

# Test 1: Query that should use tools (expect: pass)
mlflow traces evaluate \
  --scorers ToolUsageAppropriate \
  --trace-ids <trace_with_tool_use> \
  --output json

# Test 2: Greeting that shouldn't use tools (expect: pass)
mlflow traces evaluate \
  --scorers ToolUsageAppropriate \
  --trace-ids <trace_greeting> \
  --output json

# Test 3: Query that should use tools but didn't (expect: fail)
mlflow traces evaluate \
  --scorers ToolUsageAppropriate \
  --trace-ids <trace_missing_tools> \
  --output json
```

Review each output to verify scorer behaves as expected.

---

**For troubleshooting scorer issues**, see `references/troubleshooting.md`
