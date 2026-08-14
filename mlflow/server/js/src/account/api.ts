import { matchPredefinedError, UnknownError } from '@databricks/web-shared/errors';
import { fetchEndpoint } from '../common/utils/FetchUtils';
import type { ListMyPermissionsResponse, ListUserRolesResponse, UpdatePasswordRequest, UserResponse } from './types';

const defaultErrorHandler = async ({
  reject,
  response,
  err: originalError,
}: {
  reject: (cause: any) => void;
  // ``fetchEndpoint`` omits ``response`` on network/CORS/abort failures.
  response: Response | undefined;
  err: Error;
}) => {
  if (!response) {
    reject(originalError);
    return;
  }

  const predefinedError = matchPredefinedError(response);
  const error = predefinedError instanceof UnknownError ? originalError : predefinedError;
  try {
    const messageFromResponse = (await response.json())?.message;
    if (messageFromResponse) {
      error.message = messageFromResponse;
    }
  } catch {
    // Keep original error message if extraction fails
  }

  reject(error);
};

export const AccountApi = {
  getCurrentUser: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/current',
      error: defaultErrorHandler,
    }) as Promise<UserResponse>;
  },

  updatePassword: (request: UpdatePasswordRequest) => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/2.0/mlflow/users/update-password',
      method: 'PATCH',
      body: JSON.stringify(request),
      error: defaultErrorHandler,
    }) as Promise<void>;
  },

  listUserRoles: (username: string) => {
    const params = new URLSearchParams();
    params.append('username', username);
    const relativeUrl = ['ajax-api/3.0/mlflow/users/roles/list', params.toString()].join('?');
    return fetchEndpoint({
      relativeUrl,
      error: defaultErrorHandler,
    }) as Promise<ListUserRolesResponse>;
  },

  listMyPermissions: () => {
    return fetchEndpoint({
      relativeUrl: 'ajax-api/3.0/mlflow/users/current/permissions',
      error: defaultErrorHandler,
    }) as Promise<ListMyPermissionsResponse>;
  },
};
