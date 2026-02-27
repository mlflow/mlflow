import { merge } from 'lodash';
import type { ReactNode } from 'react';
import React, { createContext, useContext } from 'react';

import { shouldEnableRunEvaluationReviewUIWriteFeatures } from '../utils/FeatureUtils';

export interface GenAITracesTableConfig {
  enableRunEvaluationWriteFeatures: NonNullable<boolean | undefined>;
}

// Define a default configuration
const getDefaultConfig = (): GenAITracesTableConfig => ({
  enableRunEvaluationWriteFeatures: shouldEnableRunEvaluationReviewUIWriteFeatures() ?? false,
});

// Create the context with a default value
const GenAITracesTableConfigContext = createContext<GenAITracesTableConfig>(getDefaultConfig());

interface GenAITracesTableConfigProviderProps {
  config?: Partial<GenAITracesTableConfig>;
  children: ReactNode;
}

export const GenAITracesTableConfigProvider: React.FC<React.PropsWithChildren<GenAITracesTableConfigProviderProps>> = ({
  config = {},
  children,
}) => {
  const defaultConfig = getDefaultConfig();
  // Remove undefined values from the config object

  const mergedConfig: GenAITracesTableConfig = merge({}, defaultConfig, config);

  return (
    <GenAITracesTableConfigContext.Provider value={mergedConfig}>{children}</GenAITracesTableConfigContext.Provider>
  );
};

export const useGenAITracesTableConfig = (): GenAITracesTableConfig => {
  const context = useContext(GenAITracesTableConfigContext);

  if (!context) {
    return getDefaultConfig(); // Fallback to defaults if no provider is found
  }

  return context;
};
