/**
 * Pure presentation component for Provider and Model selection.
 *
 * Renders:
 * - Provider selector (typeahead combobox)
 * - Model selector (modal-based selection)
 * - Model capabilities display
 *
 * This component is purely presentational - all state and data
 * fetching is handled by parent components/hooks.
 */

import { useMemo, memo } from 'react';
import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import { ProviderSelect } from '../../create-endpoint/ProviderSelect';
import { ModelSelect } from '../../create-endpoint/ModelSelect';
import type { Model } from '../../../types';

export interface ProviderModelSelectorProps {
  /** Currently selected provider */
  provider: string;
  /** Currently selected model name */
  modelName: string;
  /** Callback when provider changes */
  onProviderChange: (provider: string) => void;
  /** Callback when model changes */
  onModelChange: (modelName: string) => void;
  /** Model metadata for displaying capabilities */
  modelMetadata?: Model;
  /** Provider validation error */
  providerError?: string;
  /** Model validation error */
  modelError?: string;
  /** Whether the component is disabled */
  disabled?: boolean;
  /** Component ID prefix for telemetry */
  componentIdPrefix?: string;
}

/**
 * Pure presentation component for provider and model selection
 */
export function ProviderModelSelector({
  provider,
  modelName,
  onProviderChange,
  onModelChange,
  modelMetadata,
  providerError,
  modelError,
  disabled,
  componentIdPrefix = 'mlflow.gateway.provider-model',
}: ProviderModelSelectorProps) {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <ProviderSelect
        value={provider}
        onChange={onProviderChange}
        disabled={disabled}
        error={providerError}
        componentIdPrefix={`${componentIdPrefix}.provider`}
      />
      <ModelSelect
        provider={provider}
        value={modelName}
        onChange={onModelChange}
        disabled={disabled}
        error={modelError}
        componentIdPrefix={`${componentIdPrefix}.model`}
      />
    </div>
  );
}

/**
 * Component to display model capabilities as tags
 */
export const ModelCapabilitiesTags = memo(function ModelCapabilitiesTags({
  model,
  componentIdPrefix = 'mlflow.gateway',
}: {
  model: Model | undefined;
  componentIdPrefix?: string;
}) {
  const { theme } = useDesignSystemTheme();

  const capabilities = useMemo(() => {
    const caps: string[] = [];
    if (model?.supports_function_calling) caps.push('Tools');
    if (model?.supports_reasoning) caps.push('Reasoning');
    if (model?.supports_prompt_caching) caps.push('Caching');
    return caps;
  }, [model?.supports_function_calling, model?.supports_reasoning, model?.supports_prompt_caching]);

  if (!model || capabilities.length === 0) {
    return null;
  }

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        flexWrap: 'wrap',
      }}
    >
      {capabilities.map((cap) => (
        <Tag key={cap} color="turquoise" componentId={`${componentIdPrefix}.capability.${cap.toLowerCase()}`}>
          {cap}
        </Tag>
      ))}
    </div>
  );
});
