# MLflow Judge Constraints & Requirements

Critical constraints when using `mlflow scorers register-llm-judge` CLI command.

## Table of Contents

1. [Constraint 1: {{trace}} Variable is Mutually Exclusive](#constraint-1-trace-variable-is-mutually-exclusive)
2. [Constraint 2: CLI Requires "yes"/"no" Return Values](#constraint-2-cli-requires-yesno-return-values)
3. [Constraint 3: Instructions Must Include Template Variable](#constraint-3-instructions-must-include-template-variable)

## Overview

The MLflow CLI for registering LLM judges has specific requirements. Follow these constraints to avoid registration errors.

## Constraint 1: {{trace}} Variable is Mutually Exclusive

If you use `{{trace}}` in your instructions, it MUST be the ONLY variable.

**Cannot mix {{trace}} with:**

- ❌ `{{inputs}}`
- ❌ `{{outputs}}`

**Example - Correct:**

```bash
uv run mlflow scorers register-llm-judge \
  -n "ToolUsage" \
  -i "Evaluate the trace: {{ trace }}. Did the agent use appropriate tools? Return yes or no."
```

**Example - Wrong:**

```bash
uv run mlflow scorers register-llm-judge \
  -n "ToolUsage" \
  -i "Given query {{ inputs }} and trace {{ trace }}, evaluate tools used."  # ❌ Cannot mix!
```

**Why this constraint exists:**

The `{{trace}}` variable contains everything:

- Input parameters (same as {{inputs}})
- Output responses (same as {{outputs}})
- All intermediate steps
- Tool calls
- LLM interactions

Since it includes inputs and outputs already, MLflow doesn't allow redundant variables.

**When to use {{trace}} vs {{inputs}}/{{outputs}}:**

Use `{{trace}}` when evaluating:

- ✅ Tool selection/usage
- ✅ Execution flow
- ✅ Intermediate reasoning
- ✅ Multi-step processes

Use `{{inputs}}`/`{{outputs}}` when evaluating:

- ✅ Final input/output quality only
- ✅ Response relevance
- ✅ Answer correctness

## Constraint 2: Prefer "yes"/"no" Return Values for Better UI integration

⚠️ **Use "yes"/"no" NOT "pass"/"fail"**

**Return values that are nicely integrated with the UI:**

- "yes" = criteria met
- "no" = criteria not met

**Return values that can still work but are not shown nicely in the UI:**

- "pass"/"fail"
- "true"/"false"
- "passed"/"failed"
- "1"/"0"

**Example - Good integration:**

```bash
uv run mlflow scorers register-llm-judge \
  -n "QualityCheck" \
  -i "Evaluate if {{ outputs }} is high quality. Return 'yes' if high quality, 'no' if not."
```

**Example - Still works but less nice integration:**

```bash
uv run mlflow scorers register-llm-judge \
  -n "QualityCheck" \
  -i "Evaluate if {{ outputs }} is high quality. Return 'pass' if good, 'fail' if bad."  # ❌ Wrong!
```

**Why "yes"/"no"?**

MLflow's built-in judges use the binary yes/no format and the UI is optimized for this use case. 

## Constraint 3: Instructions Must Include Template Variable

Instructions must contain at least one template variable:

- `{{ inputs }}` - Evaluation inputs
- `{{ outputs }}` - Agent outputs
- `{{ trace }}` - Complete execution trace

The above can be combined with optional variables:
- `{{ expectations }}` - Ground truth (optional)

**Example - Wrong (no variables):**

```bash
-i "Evaluate the quality. Return yes or no."  # ❌ Missing variable!
```

**Example - Correct:**

```bash
-i "Evaluate if {{ outputs }} is high quality. Return yes or no."  # ✅ Has variable
```

**Remember**: If using `{{ trace }}`, it must be the ONLY variable (see Constraint 1).

## Registration Example - All Constraints Met

```bash
# ✅ Correct - has variable, uses yes/no, correct parameters
uv run mlflow scorers register-llm-judge \
  -n "RelevanceCheck" \
  -d "Checks if response addresses the query" \
  -i "Given the response {{ outputs }}, determine if it directly addresses the query. Return 'yes' if relevant, 'no' if not."
```

```bash
# ✅ Correct - uses {{trace}} only (no other variables), yes/no, correct parameters
uv run mlflow scorers register-llm-judge \
  -n "ToolUsageCheck" \
  -d "Evaluates tool selection quality" \
  -i "Examine the trace {{ trace }}. Did the agent use appropriate tools for the query? Return 'yes' if appropriate, 'no' if not."
```

## Common Mistakes

1. **Mixing {{trace}} with {{inputs}} or {{outputs}}**

   - Error: "Cannot use trace variable with other variables"
   - Fix: Use only {{trace}} or only {{inputs}}/{{outputs}}

2. **Using "pass"/"fail" instead of "yes"/"no"**

   - Result: Scorer may not work correctly with evaluation
   - Fix: Always use "yes"/"no" format

3. **Missing template variables**

   - Error: "Instructions must contain at least one variable"
   - Fix: Include {{ outputs }}, {{ inputs }}, or {{ trace }}

4. **Wrong parameter names**
   - Check CLI help first: `mlflow scorers register-llm-judge --help`
   - Common correct parameters: `-n` (name), `-i` (instructions), `-d` (description)
