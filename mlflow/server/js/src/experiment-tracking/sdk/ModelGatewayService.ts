import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { getJson, fetchEndpoint, HTTPMethods } from '../../common/utils/FetchUtils';

export interface ModelGatewayQueryPayload {
  inputText: string;
  parameters: {
    temperature?: number;
    max_tokens?: number;
    stop?: string[];
  };
}

interface ModelGatewayResponseMetadata {
  total_tokens: number;
  completion_tokens: number;
  prompt_tokens: number;
}

export interface ModelGatewayCompletionsResponseType {
  choices: {
    text: string;
    finish_reason: string;
  }[];

  object: ModelDeploymentOutputType;

  usage: ModelGatewayResponseMetadata;
}

export interface ModelGatewayChatResponseType {
  choices: {
    message: { role: string; content: string };
    finish_reason: string;
  }[];

  object: ModelDeploymentOutputType;

  usage: ModelGatewayResponseMetadata;
}

export type ModelGatewayResponseType =
  | ModelGatewayCompletionsResponseType
  | ModelGatewayChatResponseType;

export interface ModelGatewayModelInfo {
  /**
   * "Original" name of the model (e.g. "gpt-3.5-turbo")
   */
  name: string;
  /**
   * Name of the model provider (e.g. "OpenAI")
   */
  provider: string;
}

export enum ModelGatewayRouteType {
  LLM_V1_COMPLETIONS = 'llm/v1/completions',
  LLM_V1_CHAT = 'llm/v1/chat',
  LLM_V1_EMBEDDINGS = 'llm/v1/embeddings',
}

export enum ModelDeploymentOutputType {
  LLM_V1_COMPLETIONS = 'text_completion',
  LLM_V1_CHAT = 'chat.completion',
}

/**
 * Response object for routes. Does not include model credentials.
 */
export interface ModelGatewayRoute {
  /**
   * User-defined name of the model route
   */
  name: string;
  /**
   * Type of route (e.g., embedding, text generation, etc.)
   */
  endpoint_type: ModelGatewayRouteType;
  /**
   * URL path of the route (e.g., "/gateway/completions/invocations")
   */
  endpoint_url: string;
  /**
   * Underlying ML model that can be accessed via this route. Could add other types of resources in the future.
   */
  model: ModelGatewayModelInfo;
}

export interface SearchModelGatewayRouteResponse {
  endpoints: ModelGatewayRoute[];
}

export class GatewayErrorWrapper extends ErrorWrapper {
  getGatewayErrorMessage() {
    return (
      this.textJson?.error?.message ||
      this.textJson?.message ||
      this.textJson?.toString() ||
      this.text
    );
  }
}

export const gatewayErrorHandler = ({
  reject,
  response,
  err,
}: {
  reject: (reason?: any) => void;
  response: Response;
  err: Error;
}) => {
  if (response) {
    response.text().then((text: any) => reject(new GatewayErrorWrapper(text, response.status)));
  } else if (err) {
    reject(new GatewayErrorWrapper(err, 500));
  }
};

function isGatewayResponseOfType(
  response: ModelGatewayResponseType,
  type: ModelDeploymentOutputType.LLM_V1_CHAT,
): response is ModelGatewayChatResponseType;

function isGatewayResponseOfType(
  response: ModelGatewayResponseType,
  type: ModelDeploymentOutputType.LLM_V1_COMPLETIONS,
): response is ModelGatewayCompletionsResponseType;

function isGatewayResponseOfType(
  response?: ModelGatewayResponseType,
  type?: ModelDeploymentOutputType,
) {
  return response?.object === type;
}

export class ModelGatewayService {
  static createEvaluationTextPayload(inputText: string, route: ModelGatewayRoute) {
    switch (route.endpoint_type) {
      case ModelGatewayRouteType.LLM_V1_COMPLETIONS: {
        return { prompt: inputText };
      }
      case ModelGatewayRouteType.LLM_V1_CHAT: {
        return { messages: [{ content: inputText, role: 'user' }] };
      }
      case ModelGatewayRouteType.LLM_V1_EMBEDDINGS: {
        // Should never happen
        throw new GatewayErrorWrapper(
          `Unsupported MLflow deployment endpoint type "${route.endpoint_type}"!`,
        );
      }
      default:
        throw new GatewayErrorWrapper(
          `Unknown MLflow deployment endpoint type "${route.endpoint_type}"!`,
        );
    }
  }
  static parseEvaluationResponse(response: ModelGatewayResponseType) {
    // We're supporting completions and chat responses for the time being
    if (isGatewayResponseOfType(response, ModelDeploymentOutputType.LLM_V1_COMPLETIONS)) {
      const text = response.choices[0]?.text;
      const { usage } = response;
      if (text && usage) {
        return { text, usage };
      }
    }
    if (isGatewayResponseOfType(response, ModelDeploymentOutputType.LLM_V1_CHAT)) {
      const text = response.choices[0]?.message?.content;
      const { usage } = response;
      if (text && usage) {
        return { text, usage };
      }
    }
    // Should not happen since we shouldn't call other route types for now
    throw new GatewayErrorWrapper(
      `Unrecognizable MLflow deployment response object "${response.object}"!`,
    );
  }
  /**
   * Search gateway routes
   */
  static searchModelGatewayRoutes = async (
    filter?: string,
  ): Promise<SearchModelGatewayRouteResponse> =>
    getJson({
      relativeUrl: 'ajax-api/2.0/gateway/routes',
      data: { filter },
    }) as Promise<SearchModelGatewayRouteResponse>;

  /**
   * Get gateway route
   */
  static getModelGatewayRoute = (name: string) =>
    getJson({ relativeUrl: `ajax-api/2.0/gateway/routes/${name}` }) as Promise<ModelGatewayRoute>;

  /**
   * Query a gateway route
   */
  static queryModelGatewayRoute = (route: ModelGatewayRoute, payload: ModelGatewayQueryPayload) => {
    const prefix = 'ajax-gateway';

    const { inputText } = payload;
    const textPayload = this.createEvaluationTextPayload(inputText, route);
    const data = {
      ...textPayload,
      ...payload.parameters,
    };

    return fetchEndpoint({
      relativeUrl: `${prefix}/${route.name}/invocations`,
      method: HTTPMethods.POST,
      retries: 3,
      initialDelay: 500,
      body: JSON.stringify(data),
      error: gatewayErrorHandler,
    }) as Promise<ModelGatewayResponseType>;
  };
}
