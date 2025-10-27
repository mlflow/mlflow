import React from 'react';
import { BadRequestError, InternalServerError, NotFoundError, PermissionError } from '@databricks/web-shared/errors';
import { ErrorWrapper } from './ErrorWrapper';
import { ErrorCodes } from '../constants';

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class ErrorUtils {
  static mlflowServices = {
    MODEL_REGISTRY: 'Model Registry',
    EXPERIMENTS: 'Experiments',
    MODEL_SERVING: 'Model Serving',
    RUN_TRACKING: 'Run Tracking',
  };
}

/**
 * Maps known types of ErrorWrapper (legacy) to platform's predefined error instances.
 */
export const mapErrorWrapperToPredefinedError = (errorWrapper: ErrorWrapper, requestId?: string) => {
  if (!(errorWrapper instanceof ErrorWrapper)) {
    return undefined;
  }
  const { status } = errorWrapper;
  let error: Error | undefined = undefined;
  const networkErrorDetails = { status };
  if (errorWrapper.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
    error = new NotFoundError(networkErrorDetails);
  }
  if (errorWrapper.getErrorCode() === ErrorCodes.PERMISSION_DENIED) {
    error = new PermissionError(networkErrorDetails);
  }
  if (errorWrapper.getErrorCode() === ErrorCodes.INTERNAL_ERROR) {
    error = new InternalServerError(networkErrorDetails);
  }
  if (errorWrapper.getErrorCode() === ErrorCodes.INVALID_PARAMETER_VALUE) {
    error = new BadRequestError(networkErrorDetails);
  }

  // Attempt to extract message from error wrapper and assign it to the error instance.
  const messageFromErrorWrapper = errorWrapper.getMessageField();
  if (error && messageFromErrorWrapper) {
    error.message = messageFromErrorWrapper;
  }

  return error;
};
export default ErrorUtils;
