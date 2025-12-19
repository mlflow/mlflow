import { MLFLOW_RUN_SOURCE_TYPE_TAG, MLflowRunSourceType } from '../../constants';
import type { ModelGatewayRouteType } from '../../sdk/ModelGatewayService';
import type { RunRowType } from '../experiment-page/utils/experimentPage.row-types';
import {
  extractRunRowParam,
  extractRunRowParamFloat,
  extractRunRowParamInteger,
} from '../experiment-page/utils/experimentPage.row-utils';

export const DEFAULT_PROMPTLAB_NEW_TEMPLATE_VALUE =
  'I have an online store selling {{ stock_type }}. Write a one-sentence advertisement for use in social media.';
export const DEFAULT_PROMPTLAB_INPUT_VALUES = { stock_type: 'books' };

export const DEFAULT_PROMPTLAB_OUTPUT_COLUMN = 'output';
export const DEFAULT_PROMPTLAB_PROMPT_COLUMN = 'prompt';
export const PROMPTLAB_METADATA_COLUMN_LATENCY = 'MLFLOW_latency';
export const PROMPTLAB_METADATA_COLUMN_TOTAL_TOKENS = 'MLFLOW_total_tokens';

const PARAM_MODEL_ROUTE = 'model_route';
const PARAM_ROUTE_TYPE = 'route_type';
const PARAM_PROMPT_TEMPLATE = 'prompt_template';
const PARAM_MAX_TOKENS = 'max_tokens';
const PARAM_TEMPERATURE = 'temperature';
const PARAM_STOP = 'stop';

export const extractPromptInputVariables = (promptTemplate: string) => {
  const pattern = /\{\{\s*([\w-]+)\s*\}\}/g;
  const matches = Array.from(promptTemplate.matchAll(pattern));
  if (!matches.length) {
    return [];
  }

  const uniqueMatches = new Set(matches.map(([, entry]) => entry.toLowerCase()));
  return Array.from(uniqueMatches);
};

export const getPromptInputVariableNameViolations = (promptTemplate: string) => {
  const namesWithSpacesPattern = /\{\{\s*([\w-]+(\s+[\w-]+)+)\s*\}\}/g;
  const namesWithSpacesMatch = promptTemplate.matchAll(namesWithSpacesPattern);
  const namesWithSpaces = Array.from(namesWithSpacesMatch).map(([, match]) => match);
  return { namesWithSpaces };
};

export const compilePromptInputText = (inputTemplate: string, inputValues: Record<string, string>) =>
  Object.entries(inputValues).reduce(
    (current, [key, value]) => current.replace(new RegExp(`{{\\s*${key}\\s*}}`, 'gi'), value),
    inputTemplate,
  );

/**
 * Parses the run entity and extracts its required parameters
 */
export const extractRequiredInputParamsForRun = (run: RunRowType) => {
  const promptTemplate = extractRunRowParam(run, PARAM_PROMPT_TEMPLATE);
  if (!promptTemplate) {
    return [];
  }
  const requiredInputs = extractPromptInputVariables(promptTemplate);
  return requiredInputs;
};

/**
 * Parses the run entity and extracts all information necessary for evaluating values
 */
export const extractEvaluationPrerequisitesForRun = (run: RunRowType) => {
  const routeName = extractRunRowParam(run, PARAM_MODEL_ROUTE);
  const routeType = extractRunRowParam(run, PARAM_ROUTE_TYPE) as ModelGatewayRouteType;
  const promptTemplate = extractRunRowParam(run, PARAM_PROMPT_TEMPLATE);
  const max_tokens = extractRunRowParamInteger(run, PARAM_MAX_TOKENS);

  const temperature = extractRunRowParamFloat(run, PARAM_TEMPERATURE);
  const stopString = extractRunRowParam(run, PARAM_STOP);
  const stop = stopString
    ?.slice(1, -1)
    .split(',')
    .map((item) => item.trim())
    // Remove empty entries
    .filter(Boolean);

  return { routeName, routeType, promptTemplate, parameters: { max_tokens, temperature, stop } };
};

/**
 * Returns `true` if run appears to originate from the prompt engineering,
 * thus contains necessary data for the evaluation of new values.
 */
export const canEvaluateOnRun = (run?: RunRowType) =>
  run?.tags?.[MLFLOW_RUN_SOURCE_TYPE_TAG]?.value === MLflowRunSourceType.PROMPT_ENGINEERING;
