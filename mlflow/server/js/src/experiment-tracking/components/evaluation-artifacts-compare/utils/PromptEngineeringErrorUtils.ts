import { ErrorWrapper } from '../../../../common/utils/ErrorWrapper';
import { GatewayErrorWrapper } from '../../../utils/LLMGatewayUtils';

/**
 * Due to multiple invocation methods, there are multiple error types that can be thrown.
 * This function extracts the proper error message from the error object.
 */
export const getPromptEngineeringErrorMessage = (e: GatewayErrorWrapper | ErrorWrapper | Error) => {
  const errorMessage =
    e instanceof GatewayErrorWrapper
      ? e.getGatewayErrorMessage()
      : e instanceof ErrorWrapper
      ? e.getMessageField()
      : e.message;

  return errorMessage;
};
