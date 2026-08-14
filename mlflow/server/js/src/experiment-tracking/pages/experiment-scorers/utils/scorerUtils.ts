import { ModelProvider, getModelProvider } from '../../../../gateway/utils/gatewayUtils';

/**
 * Determines whether the user should be allowed to toggle automatic evaluation.
 * On DB, auto-eval is always allowed since all models use gateway endpoints.
 * On OSS, auto-eval is only allowed for gateway models.
 */
export const isAutoEvaluationSupported = (model: string | undefined, hasExpectations: boolean): boolean => {
  if (hasExpectations) {
    return false;
  }
  return getModelProvider(model) !== ModelProvider.OTHER;
};
