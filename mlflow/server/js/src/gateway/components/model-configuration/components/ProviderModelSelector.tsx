import { useMemo, memo } from 'react';
import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import { NavigableProviderSelect as ProviderSelect } from '../../create-endpoint/NavigableProviderSelect';
import { ModelSelect } from '../../create-endpoint/ModelSelect';
import type { Model } from '../../../types';

export interface ProviderModelSelectorProps {
  provider: string;
  modelName: string;
  onProviderChange: (provider: string) => void;
  onModelChange: (modelName: string) => void;
  modelMetadata?: Model;
  providerError?: string;
  modelError?: string;
  disabled?: boolean;
  componentIdPrefix?: string;
}

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
