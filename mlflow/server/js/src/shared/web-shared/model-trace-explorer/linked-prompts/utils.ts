import { createMLflowRoutePath, generatePath } from '../RoutingUtils';

export const MLFLOW_LINKED_PROMPTS_TAG = 'mlflow.linkedPrompts';

const PROMPT_VERSION_QUERY_PARAM = 'promptVersion';

export interface LinkedPrompt {
  name: string;
  version: string;
}

export interface GetLinkedPromptRouteParams extends LinkedPrompt {
  experimentId: string;
}

export const getLinkedPromptRoute = ({ experimentId, name, version }: GetLinkedPromptRouteParams): string => {
  const route = generatePath(createMLflowRoutePath('/experiments/:experimentId/prompts/:promptName'), {
    experimentId,
    promptName: encodeURIComponent(name),
  });

  if (!version) {
    return route;
  }

  const searchParams = new URLSearchParams({ [PROMPT_VERSION_QUERY_PARAM]: version });
  return `${route}?${searchParams.toString()}`;
};

export const parseLinkedPrompts = (value?: string): LinkedPrompt[] => {
  try {
    const parsedValue: unknown = JSON.parse(value ?? '[]');
    if (!Array.isArray(parsedValue)) {
      return [];
    }
    return parsedValue.filter(
      (prompt): prompt is LinkedPrompt =>
        typeof prompt === 'object' &&
        prompt !== null &&
        'name' in prompt &&
        typeof prompt.name === 'string' &&
        'version' in prompt &&
        typeof prompt.version === 'string',
    );
  } catch {
    return [];
  }
};
