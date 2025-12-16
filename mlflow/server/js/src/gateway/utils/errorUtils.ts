/**
 * Check if error message indicates a unique constraint violation across different DB backends
 */
export const isUniqueConstraintError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return (
    // SQLite
    lowerMessage.includes('unique constraint failed') ||
    // PostgreSQL
    lowerMessage.includes('duplicate key value violates unique constraint') ||
    // MySQL
    lowerMessage.includes('duplicate entry') ||
    // SQL Server
    lowerMessage.includes('violation of unique key constraint') ||
    // Generic patterns
    lowerMessage.includes('uniqueviolation') ||
    lowerMessage.includes('integrityerror') ||
    // User-friendly message from backend
    lowerMessage.includes('already exists')
  );
};

/**
 * Check if the error is related to an endpoint name
 */
export const isEndpointNameError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return lowerMessage.includes('endpoints.name') || lowerMessage.includes('endpoint_name');
};

/**
 * Check if the error is related to a secret name
 */
export const isSecretNameError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return (
    lowerMessage.includes('secrets.secret_name') ||
    lowerMessage.includes('secret_name') ||
    lowerMessage.includes('secret with name')
  );
};

/**
 * Check if an error is a secret name conflict
 */
export const isSecretNameConflict = (error: unknown): boolean => {
  const errorMessage = (error as Error)?.message ?? '';
  return isUniqueConstraintError(errorMessage) && isSecretNameError(errorMessage);
};

export interface GetReadableErrorMessageOptions {
  entityType?: 'endpoint' | 'secret';
  action?: 'creating' | 'updating' | 'deleting';
}

/**
 * Parse backend error messages and return user-friendly versions
 */
export const getReadableErrorMessage = (
  error: Error | null,
  options: GetReadableErrorMessageOptions = {},
): string | null => {
  if (!error?.message) return null;

  const { entityType, action = 'creating' } = options;
  const message = error.message;

  if (isUniqueConstraintError(message)) {
    if (isEndpointNameError(message)) {
      return 'An endpoint with this name already exists. Please choose a different name.';
    }
    if (isSecretNameError(message)) {
      return entityType === 'secret'
        ? 'A secret with this name already exists. Please choose a different name or use an existing secret.'
        : 'An API key with this name already exists. Please choose a different name or use an existing API key.';
    }
    // Generic unique constraint fallback
    return 'A record with this value already exists. Please use a unique value.';
  }

  // Return original message if no pattern matched (but truncate if too long)
  if (message.length > 200) {
    const actionText = action === 'updating' ? 'updating' : action === 'deleting' ? 'deleting' : 'creating';
    const entityText = entityType === 'secret' ? 'secret' : entityType === 'endpoint' ? 'endpoint' : 'resource';
    return `An error occurred while ${actionText} the ${entityText}. Please try again.`;
  }

  return message;
};
