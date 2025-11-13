import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import { extractInputs, extractOutputs, extractExpectations } from './TraceUtils';

/**
 * Constants for judge evaluation prompts.
 *
 * These constants mirror the Python implementation in:
 * mlflow/genai/judges/instructions_judge/constants.py
 * and mlflow/genai/judges/instructions_judge/__init__.py
 */

// Common base prompt for all judge evaluations
const JUDGE_BASE_PROMPT = `You are an expert judge tasked with evaluating the performance of an AI
agent on a particular query. You will be given instructions that describe the criteria and
methodology for evaluating the agent's performance on the query.`;

// Output format instructions for judge responses
const OUTPUT_FORMAT_INSTRUCTIONS = `
Please provide your assessment in the following JSON format only (no markdown):

{
    "result": "The evaluation rating/result",
    "rationale": "Detailed explanation for the evaluation"
}`;

// Fallback message when no data is provided
const FALLBACK_USER_MESSAGE = 'Follow the instructions from the first message';

// Pattern for extracting template variables like {{inputs}}, {{outputs}}, {{expectations}}
// Matches Python pattern: mlflow/prompt/constants.py PROMPT_TEMPLATE_VARIABLE_PATTERN
const TEMPLATE_VARIABLE_PATTERN = /\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s*\}\}/g;

// Reserved template variables that can be used in instructions
export const TEMPLATE_VARIABLES = ['inputs', 'outputs', 'expectations', 'trace'];

/**
 * Extracts template variables from instructions string. Variables are returned in the order they first appear.
 *
 * Example: "Evaluate if {{outputs}} correctly answers {{inputs}}" returns ['outputs', 'inputs']
 *
 * @param instructions - The judge instructions string with template variables
 * @returns Array of variable names found in the instructions
 */
export function extractTemplateVariables(instructions: string): string[] {
  const matches = instructions.matchAll(TEMPLATE_VARIABLE_PATTERN);
  const seen = new Set<string>();
  const variables: string[] = [];

  for (const match of matches) {
    const varName = match[1];
    if (!seen.has(varName) && TEMPLATE_VARIABLES.includes(varName)) {
      seen.add(varName);
      variables.push(varName);
    }
  }

  return variables;
}

/**
 * Combined extraction function for all trace data
 */
export function extractFromTrace(trace: ModelTrace): {
  inputs: string | null;
  outputs: string | null;
  expectations: Record<string, any>;
} {
  return {
    inputs: extractInputs(trace),
    outputs: extractOutputs(trace),
    expectations: extractExpectations(trace),
  };
}

/**
 * Builds the system prompt for the judge
 */
export function buildSystemPrompt(instructions: string): string {
  const taskSection = `Your task: ${instructions}.`;
  return `${JUDGE_BASE_PROMPT}\n\n${taskSection}\n${OUTPUT_FORMAT_INSTRUCTIONS}`;
}

/**
 * Builds the user prompt with extracted data.
 *
 * Only includes fields that are referenced in the instructions template variables.
 *
 * @param inputs - Extracted inputs from trace
 * @param outputs - Extracted outputs from trace
 * @param expectations - Extracted expectations from trace
 * @param templateVariables - Variables found in the instructions
 * @returns Formatted user prompt string
 */
export function buildUserPrompt(
  inputs: string | null,
  outputs: string | null,
  expectations: Record<string, any>,
  templateVariables: string[],
): string {
  const parts: string[] = [];

  // Build a map of variable name to data
  const dataMap: Record<string, any> = {
    inputs,
    outputs,
    expectations,
  };

  // Only include variables that are in the template AND have non-null/non-empty data
  for (const varName of templateVariables) {
    const data = dataMap[varName];

    if (varName === 'inputs' && data !== null && data !== undefined && data !== '') {
      parts.push(`inputs: ${data}`);
    } else if (varName === 'outputs' && data !== null && data !== undefined && data !== '') {
      parts.push(`outputs: ${data}`);
    } else if (varName === 'expectations' && data && Object.keys(data).length > 0) {
      parts.push(`expectations: ${JSON.stringify(data, null, 2)}`);
    }
  }

  // Some model providers require non-empty user messages
  return parts.length > 0 ? parts.join('\n') : FALLBACK_USER_MESSAGE;
}
