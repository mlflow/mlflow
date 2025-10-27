import { FormattedMessage } from 'react-intl';

// eslint-disable-next-line no-restricted-imports
import type { ServerError } from '@apollo/client';

import { ErrorLogType } from './ErrorLogType';
import { ErrorName } from './ErrorName';

export type HandleableError = Error | string | Record<string, unknown> | PredefinedError | Response;

export type CausableError = Error | string | Record<string, unknown>;

export abstract class PredefinedError extends Error {
  abstract errorLogType: ErrorLogType;
  abstract errorName: ErrorName;
  abstract displayMessage: React.ReactNode;
  isUserError = false;

  constructor(message?: string, cause?: CausableError) {
    super(message);
  }
}

export const matchPredefinedError = (error: HandleableError) => {
  if (error instanceof PredefinedError) {
    return error;
  }
  if (error instanceof Error && ('networkError' in error || 'graphQLErrors' in error)) {
    return matchPredefinedApolloError(error);
  }

  if (error instanceof Response) {
    return matchPredefinedErrorFromResponse(error);
  }

  return new UnknownError(error);
};

export function isServerError(e: unknown): e is ServerError {
  return e instanceof Error && e.hasOwnProperty('response');
}

const matchPredefinedApolloError = (error: Error) => {
  // Some errors from Apollo mock provider may have `networkError` but are not `ServerError`
  // only act on ServerError, which do have the response attached
  if ('networkError' in error && isServerError(error.networkError)) {
    return matchPredefinedErrorFromResponse(error.networkError.response, error.networkError);
  }

  return new GraphQLGenericError(error);
};

const getNetworkRequestErrorDetailsFromResponse = (response: Response): NetworkRequestErrorDetails => {
  const status = response.status;

  return { status };
};

export const matchPredefinedErrorFromResponse = (response: Response, originalError?: CausableError) => {
  const errorDetails = NetworkRequestError.getNetworkRequestErrorDetailsFromResponse(response);
  switch (response.status) {
    case 400:
      return new BadRequestError(errorDetails, originalError);
    case 401:
      return new UnauthorizedError(errorDetails, originalError);
    case 403:
      return new PermissionError(errorDetails, originalError);
    case 404:
      return new NotFoundError(errorDetails, originalError);
    case 429:
      return new RateLimitedError(errorDetails, originalError);
    case 500:
      return new InternalServerError(errorDetails, originalError);
    case 503:
      return new ServiceUnavailableError(errorDetails, originalError);
    default:
      return new GenericNetworkRequestError(errorDetails, originalError);
  }
};

interface NetworkRequestErrorDetails {
  status?: number;
  response?: Response;
}

export abstract class NetworkRequestError extends PredefinedError {
  status?: number;
  response?: Response;

  constructor(message: string, details: NetworkRequestErrorDetails, cause?: CausableError) {
    super(message, cause);
    this.status = details.status;
    this.response = details.response;
  }

  static getNetworkRequestErrorDetailsFromResponse = getNetworkRequestErrorDetailsFromResponse;
}

export class GenericNetworkRequestError extends NetworkRequestError {
  errorLogType = ErrorLogType.ServerError;
  errorName = ErrorName.GenericNetworkRequestError;
  displayMessage = (
    <FormattedMessage defaultMessage="A network error occurred." description="Generic message for a network error" />
  );

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'A network error occurred.';

    super(message, details, cause);
  }
}

export class GraphQLGenericError extends PredefinedError {
  errorLogType = ErrorLogType.ApplicationError;
  errorName = ErrorName.GraphQLGenericError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="A GraphQL error occurred."
      description="Generic message for a GraphQL error, typically due to query parsing or validation issues"
    />
  );

  constructor(cause?: CausableError) {
    const message = 'A GraphQL error occurred.';

    super(message, cause);
  }
}

export class BadRequestError extends NetworkRequestError {
  errorLogType = ErrorLogType.UserInputError;
  errorName = ErrorName.BadRequestError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="The request was invalid."
      description="Bad request (HTTP STATUS 400) generic error message"
    />
  );

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'The request was invalid.';

    super(message, details, cause);
  }
}

export class InternalServerError extends NetworkRequestError {
  errorLogType = ErrorLogType.ServerError;
  errorName = ErrorName.InternalServerError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="Internal server error"
      description="Request failed due to internal server error (HTTP STATUS 500) generic error message"
    />
  );

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'Internal server error';

    super(message, details, cause);
  }
}

export class NotFoundError extends NetworkRequestError {
  errorLogType = ErrorLogType.UserInputError;
  errorName = ErrorName.NotFoundError;

  displayMessage = (
    <FormattedMessage
      defaultMessage="The requested resource was not found."
      description="Resource not found (HTTP STATUS 404) generic error message"
    />
  );

  isUserError = true;

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'The requested resource was not found.';

    super(message, details, cause);
  }
}

export class PermissionError extends NetworkRequestError {
  errorLogType = ErrorLogType.UnexpectedSystemStateError;
  errorName = ErrorName.PermissionError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="You do not have permission to access this resource."
      description="Generic message for a permission error (HTTP STATUS 403)"
    />
  );
  isUserError = true;

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'You do not have permission to access this resource.';

    super(message, details, cause);
  }
}

export class RateLimitedError extends NetworkRequestError {
  errorLogType = ErrorLogType.ServerError;
  errorName = ErrorName.RateLimitedError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="This request exceeds the maximum queries per second limit. Please wait and try again."
      description="Too many requests (HTTP STATUS 429) generic error message"
    />
  );

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'This request exceeds the maximum queries per second limit. Please wait and try again.';

    super(message, details, cause);
  }
}

export class ServiceUnavailableError extends NetworkRequestError {
  errorLogType = ErrorLogType.ServerError;
  errorName = ErrorName.InternalServerError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="Service unavailable error"
      description="Request failed due to service being available (HTTP STATUS 503) generic error message"
    />
  );

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'Internal server error';

    super(message, details, cause);
  }
}

export class UnauthorizedError extends NetworkRequestError {
  errorLogType = ErrorLogType.SessionError;
  errorName = ErrorName.UnauthorizedError;
  displayMessage = (
    <FormattedMessage
      defaultMessage="User is not authorized."
      description="Unauthorized (HTTP STATUS 401) generic error message"
    />
  );

  constructor(details: NetworkRequestErrorDetails, cause?: CausableError) {
    const message = 'This request exceeds the maximum queries per second limit. Please wait and try again.';

    super(message, details, cause);
  }
}

export class UnknownError extends PredefinedError {
  errorLogType = ErrorLogType.UnknownError;
  errorName = ErrorName.UnknownError;
  displayMessage = (
    <FormattedMessage defaultMessage="An unknown error occurred." description="Generic message for an unknown error" />
  );

  constructor(cause?: CausableError) {
    const message = 'An unknown error occurred.';

    super(message, cause);
  }
}

export class FormValidationError extends PredefinedError {
  errorLogType = ErrorLogType.UserInputError;
  errorName = ErrorName.FormValidationError;
  isUserError = true;
  displayMessage = (
    <FormattedMessage
      defaultMessage="At least one form field has incorrect value. Please correct and try again."
      description="Generic error message for an invalid form input"
    />
  );

  constructor(cause?: CausableError) {
    const message = 'Incorrect form input.';

    super(message, cause);
  }
}

// have to be defined here to avoid circular dependencies
export class RouteNotFoundError extends PredefinedError {
  errorLogType = ErrorLogType.UserInputError;
  errorName = ErrorName.RouteNotFoundError;
  isUserError = true;
  displayMessage = (
    <FormattedMessage
      defaultMessage="Page not found"
      description="Error message shown to the user when they arrive at a non existent URL"
    />
  );
  constructor() {
    super('Page not found');
  }
}
