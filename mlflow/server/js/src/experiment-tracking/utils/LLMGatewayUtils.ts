import { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { ModelGatewayRouteTask } from '../sdk/MlflowEnums';
import type {
  EndpointModelChatResponseType,
  EndpointModelCompletionsResponseType,
  EndpointModelGatewayResponseType,
  ModelGatewayChatResponseType,
  ModelGatewayCompletionsResponseType,
  ModelGatewayResponseType,
  ModelGatewayRoute,
} from '../sdk/ModelGatewayService';

export class GatewayErrorWrapper extends ErrorWrapper {
  getGatewayErrorMessage() {
    return this.textJson?.error?.message || this.textJson?.message || this.textJson?.toString() || this.text;
  }
}
export const parseEndpointEvaluationResponse = (
  response: EndpointModelGatewayResponseType,
  task: ModelGatewayRouteTask,
) => {
  // We're supporting completions and chat responses for the time being
  if (task === ModelGatewayRouteTask.LLM_V1_COMPLETIONS) {
    const completionsResponse = response as EndpointModelCompletionsResponseType;
    const text = completionsResponse.choices?.[0]?.text;
    const { usage } = completionsResponse;
    if (text && usage) {
      return {
        text,
        metadata: {
          total_tokens: usage.total_tokens,
          output_tokens: usage.completion_tokens,
          input_tokens: usage.prompt_tokens,
        },
      };
    }
  }
  if (task === ModelGatewayRouteTask.LLM_V1_CHAT) {
    const chatResponse = response as EndpointModelChatResponseType;
    const text = chatResponse.choices?.[0]?.message?.content;
    const { usage } = chatResponse;
    if (text && usage) {
      return {
        text,
        metadata: {
          total_tokens: usage.total_tokens,
          output_tokens: usage.completion_tokens,
          input_tokens: usage.prompt_tokens,
        },
      };
    }
  }
  // Should not happen since we shouldn't call other route types for now
  throw new Error(`Unrecognizable AI gateway response metadata "${response.usage}"!`);
};
