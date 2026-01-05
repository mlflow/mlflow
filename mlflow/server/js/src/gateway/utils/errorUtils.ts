const isUniqueConstraintError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return (
    lowerMessage.includes('unique constraint failed') ||
    lowerMessage.includes('duplicate key value violates unique constraint') ||
    lowerMessage.includes('duplicate entry') ||
    lowerMessage.includes('violation of unique key constraint') ||
    lowerMessage.includes('uniqueviolation') ||
    lowerMessage.includes('integrityerror') ||
    lowerMessage.includes('already exists')
  );
};

const isEndpointNameError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return lowerMessage.includes('endpoints.name') || lowerMessage.includes('endpoint_name');
};

const isSecretNameError = (message: string): boolean => {
  const lowerMessage = message.toLowerCase();
  return (
    lowerMessage.includes('secrets.secret_name') ||
    lowerMessage.includes('secret_name') ||
    lowerMessage.includes('secret with name')
  );
};

export const isSecretNameConflict = (error: unknown): boolean => {
  const errorMessage = (error as Error)?.message ?? '';
  return isUniqueConstraintError(errorMessage) && isSecretNameError(errorMessage);
};

interface GetReadableErrorMessageOptions {
  entityType?: 'endpoint' | 'secret';
  action?: 'creating' | 'updating' | 'deleting';
}

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
    return 'A record with this value already exists. Please use a unique value.';
  }

  if (message.length > 200) {
    const actionText = action === 'updating' ? 'updating' : action === 'deleting' ? 'deleting' : 'creating';
    const entityText = entityType === 'secret' ? 'secret' : entityType === 'endpoint' ? 'endpoint' : 'resource';
    return `An error occurred while ${actionText} the ${entityText}. Please try again.`;
  }

  return message;
};
