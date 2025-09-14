import invariant from 'invariant';
import { getJson } from '../../common/utils/FetchUtils';
import { MlflowService } from './MlflowService';
import { ModelGatewayRouteTask } from './MlflowEnums';
import { GatewayErrorWrapper } from '../utils/LLMGatewayUtils';
import { fetchEndpoint, HTTPMethods } from '../../common/utils/FetchUtils';
import { parseEndpointEvaluationResponse } from '../utils/LLMGatewayUtils';
const DATABRICKS_API_CLIENT_PROMPTLAB = 'PromptLab';

export interface ModelGatewayQueryPayload {
  inputText: string;
  parameters: {
    temperature?: number;
    max_tokens?: number;
    stop?: string[];
  };
}

export interface ModelGatewayResponseMetadata<T extends ModelGatewayRouteTask> {
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

  metadata: ModelGatewayResponseMetadata<ModelGatewayRouteTask.LLM_V1_COMPLETIONS>;
}

export interface ModelGatewayChatResponseType {
  candidates: {
    message: { role: string; content: string };
    metadata: {
      finish_reason: string;
    };
  }[];

  metadata: ModelGatewayResponseMetadata<ModelGatewayRouteTask.LLM_V1_CHAT>;
}

export type ModelGatewayResponseType = ModelGatewayCompletionsResponseType | ModelGatewayChatResponseType;

export interface EndpointModelCompletionsResponseType {
  choices: {
    text: string;
    finish_reason: string;
  }[];

  usage: {
    completion_tokens: number;
    prompt_tokens: number;
    total_tokens: number;
  };
}

export interface EndpointModelChatResponseType {
  choices: {
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }[];

  usage: {
    completion_tokens: number;
    prompt_tokens: number;
    total_tokens: number;
  };
}

export type EndpointModelGatewayResponseType = EndpointModelCompletionsResponseType | EndpointModelChatResponseType;

export interface ModelGatewayModelInfo {
  /**
   * "Original" name of the model (e.g. "gpt-4o-mini")
   */
  name: string;
  /**
   * Name of the model provider (e.g. "OpenAI")
   */
  provider: string;
}

/**
 * Response object for routes. Does not include model credentials.
 */
export interface ModelGatewayRouteLegacy {
  /**
   * User-defined name of the model route
   */
  name: string;
  /**
   * Type of route (e.g., embedding, text generation, etc.)
   */
  route_type: ModelGatewayRouteTask;
  /**
   * Underlying ML model that can be accessed via this route. Could add other types of resources in the future.
   */
  model: ModelGatewayModelInfo;
}

export interface MlflowDeploymentsEndpoint {
  name: string;
  endpoint_type: ModelGatewayRouteTask;
  endpoint_url: string;
  model: ModelGatewayModelInfo;
}

export type ModelGatewayRouteType = 'mlflow_deployment_endpoint';

export interface ModelGatewayRoute {
  type: ModelGatewayRouteType;
  /**
   * Key of the route, the type is always prefix
   */
  key: `${ModelGatewayRouteType}:${string}`;

  name: string;
  /**
   * Type of route (e.g., embedding, text generation, etc.)
   */
  task: ModelGatewayRouteTask;
  /**
   * MLflow deployments URL of the endpoint
   */
  mlflowDeployment?: MlflowDeploymentsEndpoint;
}

export interface SearchMlflowDeploymentsModelRoutesResponse {
  endpoints: MlflowDeploymentsEndpoint[];
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

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class ModelGatewayService {
  static createEvaluationTextPayload(inputText: string, task: ModelGatewayRouteTask) {
    switch (task) {
      case ModelGatewayRouteTask.LLM_V1_COMPLETIONS: {
        return { prompt: inputText };
      }
      case ModelGatewayRouteTask.LLM_V1_CHAT: {
        return { messages: [{ content: inputText, role: 'user' }] };
      }
      case ModelGatewayRouteTask.LLM_V1_EMBEDDINGS: {
        // Should never happen
        throw new Error(`Unsupported served LLM model task "${task}"!`);
      }
      default:
        throw new Error(`Unknown served LLM model task "${task}"!`);
    }
  }

  static queryMLflowDeploymentEndpointRoute = async (
    route: ModelGatewayRoute,
    data: ModelGatewayQueryPayload,
  ): Promise<any> => {
    invariant(route.mlflowDeployment, 'Trying to call a MLflow deployment route without a deployment_url');
    const { inputText } = data;
    const textPayload = ModelGatewayService.createEvaluationTextPayload(inputText, route.task);
    const processed_data = {
      ...textPayload,
      ...data.parameters,
    };

    return MlflowService.gatewayProxyPost({
      gateway_path: route.mlflowDeployment.endpoint_url.substring(1),
      json_data: processed_data,
    }) as Promise<ModelGatewayResponseType>;
  };

  static queryModelGatewayRoute = async (route: ModelGatewayRoute, payload: ModelGatewayQueryPayload) => {
    if (route.type === 'mlflow_deployment_endpoint') {
      invariant(route.mlflowDeployment, 'Trying to call a serving endpoint route without an endpoint');
      const result = await this.queryMLflowDeploymentEndpointRoute(route, payload);
      return parseEndpointEvaluationResponse(result, route.task);
    }

    throw new Error('Unknown route type');
  };
}
