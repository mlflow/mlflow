# Agent Evaluation Report

**Agent**: [Agent Name]
**Date**: [YYYY-MM-DD]
**Evaluator**: [Name]
**Dataset**: [Dataset Name] ([N] queries)

## Executive Summary

[1-2 paragraph summary of key findings, overall performance, and main recommendations]

## Evaluation Setup

**Configuration**:

- **Tracking URI**: [URI]
- **Experiment ID**: [ID]
- **Dataset**: [Name] ([N] queries)
- **MLflow Version**: [version]

**Scorers Used**:

1. **[Scorer 1 Name]**: [Brief description of what it evaluates]
2. **[Scorer 2 Name]**: [Brief description]
3. **[Scorer 3 Name]**: [Brief description]
4. ...

## Results Overview

### Overall Performance

| Scorer              | Pass Rate | Passed/Total | Grade       |
| ------------------- | --------- | ------------ | ----------- |
| [Scorer 1]          | [X]%      | [Y]/[Z]      | [A/B/C/F]   |
| [Scorer 2]          | [X]%      | [Y]/[Z]      | [A/B/C/F]   |
| [Scorer 3]          | [X]%      | [Y]/[Z]      | [A/B/C/F]   |
| **Overall Average** | **[X]%**  | **-**        | **[Grade]** |

**Grading Scale**: A (90-100%), B (80-89%), C (70-79%), D (60-69%), F (<60%)

### Performance Distribution

```
█████████░░ 90%  [Scorer 1]
███████░░░░ 70%  [Scorer 2]
████████░░░ 80%  [Scorer 3]
```

## Detailed Findings

### [Scorer 1 Name]

**Performance**: [Pass rate]% ([Passed]/[Total])
**Grade**: [A/B/C/D/F]

**Strengths**:

- [What worked well - specific examples]
- [Another strength]

**Issues**:

- [What didn't work - specific examples]
- [Another issue]

**Example Failures**:

1. **Query**: "[failing query]"

   - **Issue**: [Why it failed]
   - **Trace**: [trace_id or link]

2. **Query**: "[another failing query]"
   - **Issue**: [Why it failed]
   - **Trace**: [trace_id or link]

### [Scorer 2 Name]

[Same structure as above]

### [Scorer 3 Name]

[Same structure as above]

## Failure Pattern Analysis

### Pattern 1: [Pattern Name]

**Description**: [What is the pattern]

**Frequency**: [N queries affected] ([X]% of failures)

**Affected Queries**:

- "[example query 1]"
- "[example query 2]"
- "[example query 3]"

**Root Cause**: [Why this pattern occurs]

**Impact**: [Severity/importance]

### Pattern 2: [Pattern Name]

[Same structure as above]

### Pattern 3: [Pattern Name]

[Same structure as above]

## Recommendations

### Immediate Actions (High Priority)

1. **[Action Item 1]**

   - **Issue**: [What problem this addresses]
   - **Expected Impact**: [What will improve]
   - **Effort**: [Low/Medium/High]

2. **[Action Item 2]**
   - **Issue**: [What problem this addresses]
   - **Expected Impact**: [What will improve]
   - **Effort**: [Low/Medium/High]

### Short-Term Improvements (Medium Priority)

1. **[Action Item 3]**

   - **Issue**: [What problem this addresses]
   - **Expected Impact**: [What will improve]
   - **Effort**: [Low/Medium/High]

2. **[Action Item 4]**
   - **Issue**: [What problem this addresses]
   - **Expected Impact**: [What will improve]
   - **Effort**: [Low/Medium/High]

### Long-Term Enhancements (Low Priority)

1. **[Enhancement 1]**: [Description and expected impact]
2. **Enhancement 2]**: [Description and expected impact]

## Dataset Analysis

**Size**: [N] queries
**Diversity**: [High/Medium/Low]

**Query Distribution**:

- Short queries (<10 words): [N] ([X]%)
- Medium queries (10-20 words): [N] ([X]%)
- Long queries (>20 words): [N] ([X]%)

**Coverage Assessment**:

- ✓ [Covered capability 1]
- ✓ [Covered capability 2]
- ✗ [Missing capability 1] - **Consider adding queries for this**
- ✗ [Missing capability 2] - **Consider adding queries for this**

## Next Steps

1. **Address immediate actions** listed above
2. **Re-evaluate** after implementing fixes
3. **Expand dataset** to cover identified gaps
4. **Monitor production** for similar failure patterns

## Appendix

### Evaluation Run Details

**Run ID**: [mlflow_run_id]
**Run Name**: [run_name]
**Timestamp**: [timestamp]
**Duration**: [execution_time]

### Evaluation Command

```bash
mlflow traces evaluate \
  --trace-ids [comma_separated_trace_ids] \
  --scorers [comma_separated_scorers] \
  --output json
```

### All Trace IDs

```
[trace_id_1]
[trace_id_2]
[trace_id_3]
...
```

### Environment

- **Python Version**: [version]
- **MLflow Version**: [version]
- **Agent Library**: [library and version]
- **LLM Model**: [model used by agent]

---

**Report Generated**: [timestamp]
**Evaluation Framework**: MLflow Agent Evaluation
