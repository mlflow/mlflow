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

interface ModelGatewayResponseMetadata<T extends ModelGatewayRouteType> {
  mode: string;
  route_type: T;
  total_tokens: number;
  output_tokens: number;
  input_tokens: number;
}

export interface ModelGatewayCompletionsResponseType {
  candidates: {
    text: string;
    metadata: {
      finish_reason: string;
    };
  }[];

  metadata: ModelGatewayResponseMetadata<ModelGatewayRouteType.LLM_V1_COMPLETIONS>;
}

export interface ModelGatewayChatResponseType {
  candidates: {
    message: { role: string; content: string };
    metadata: {
      finish_reason: string;
    };
  }[];

  metadata: ModelGatewayResponseMetadata<ModelGatewayRouteType.LLM_V1_CHAT>;
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
  route_type: ModelGatewayRouteType;
  /**
   * Underlying ML model that can be accessed via this route. Could add other types of resources in the future.
   */
  model: ModelGatewayModelInfo;
}

export interface SearchModelGatewayRouteResponse {
  routes: ModelGatewayRoute[];
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

const gatewayErrorHandler = ({
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
  type: ModelGatewayRouteType.LLM_V1_CHAT,
): response is ModelGatewayChatResponseType;

function isGatewayResponseOfType(
  response: ModelGatewayResponseType,
  type: ModelGatewayRouteType.LLM_V1_COMPLETIONS,
): response is ModelGatewayCompletionsResponseType;

function isGatewayResponseOfType(
  response?: ModelGatewayResponseType,
  type?: ModelGatewayRouteType,
) {
  return response?.metadata.route_type === type;
}

export class ModelGatewayService {
  static createEvaluationTextPayload(inputText: string, route: ModelGatewayRoute) {
    switch (route.route_type) {
      case ModelGatewayRouteType.LLM_V1_COMPLETIONS: {
        return { prompt: inputText };
      }
      case ModelGatewayRouteType.LLM_V1_CHAT: {
        return { messages: [{ content: inputText, role: 'user' }] };
      }
      case ModelGatewayRouteType.LLM_V1_EMBEDDINGS: {
        // Should never happen
        throw new GatewayErrorWrapper(`Unsupported AI gateway route type "${route.route_type}"!`);
      }
      default:
        throw new GatewayErrorWrapper(`Unknown AI gateway route type "${route.route_type}"!`);
    }
  }
  static parseEvaluationResponse(response: ModelGatewayResponseType) {
    // We're supporting completions and chat responses for the time being
    if (isGatewayResponseOfType(response, ModelGatewayRouteType.LLM_V1_COMPLETIONS)) {
      const text = response.candidates[0]?.text;
      const { metadata } = response;
      if (text && metadata) {
        return { text, metadata };
      }
    }
    if (isGatewayResponseOfType(response, ModelGatewayRouteType.LLM_V1_CHAT)) {
      const text = response.candidates[0]?.message?.content;
      const { metadata } = response;
      if (text && metadata) {
        return { text, metadata };
      }
    }
    // Should not happen since we shouldn't call other route types for now
    throw new GatewayErrorWrapper(
      `Unrecognizable AI gateway response metadata "${response.metadata}"!`,
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
