import { useState, useMemo, useCallback, memo } from 'react';
import { Input, useDesignSystemTheme, FormUI, Tag, ModelsIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ModelSelectorModal } from '../model-selector/ModelSelectorModal';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import type { ProviderModel } from '../../types';

interface ModelSelectProps {
  provider: string;
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
  error?: string;
  /** Component ID prefix for telemetry (default: 'mlflow.gateway.model-select') */
  componentIdPrefix?: string;
}

export const ModelSelect = ({
  provider,
  value,
  onChange,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.model-select',
}: ModelSelectProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Fetch models to get the selected model's details
  const { data: models } = useModelsQuery({ provider: provider || undefined });
  const selectedModel = useMemo(() => models?.find((m) => m.model === value), [models, value]);

  const handleClick = useCallback(() => {
    if (!disabled && provider) {
      setIsModalOpen(true);
    }
  }, [disabled, provider]);

  const handleSelect = useCallback(
    (model: ProviderModel) => {
      onChange(model.model);
    },
    [onChange],
  );

  const handleClose = useCallback(() => {
    setIsModalOpen(false);
  }, []);

  return (
    <div>
      <FormUI.Label htmlFor={componentIdPrefix}>
        <FormattedMessage defaultMessage="Model" description="Label for model select field" />
      </FormUI.Label>
      <Input
        id={componentIdPrefix}
        componentId={componentIdPrefix}
        placeholder={
          !provider
            ? intl.formatMessage({
                defaultMessage: 'Select a provider first',
                description: 'Placeholder when no provider selected',
              })
            : intl.formatMessage({
                defaultMessage: 'Click to select a model',
                description: 'Placeholder for model selection',
              })
        }
        readOnly
        disabled={disabled || !provider}
        onClick={handleClick}
        value={value || ''}
        validationState={error ? 'error' : undefined}
        prefix={value ? <ModelsIcon /> : undefined}
        css={{
          cursor: disabled || !provider ? 'not-allowed' : 'pointer',
          '& input': {
            cursor: disabled || !provider ? 'not-allowed' : 'pointer',
          },
        }}
      />
      {error && <FormUI.Message type="error" message={error} />}
      {selectedModel && <ModelCapabilities model={selectedModel} />}
      <ModelSelectorModal isOpen={isModalOpen} onClose={handleClose} onSelect={handleSelect} provider={provider} />
    </div>
  );
};

const ModelCapabilities = memo(({ model }: { model: ProviderModel }) => {
  const { theme } = useDesignSystemTheme();

  const capabilities = useMemo(() => {
    const caps: string[] = [];
    if (model.supports_function_calling) caps.push('Tools');
    if (model.supports_reasoning) caps.push('Reasoning');
    if (model.supports_prompt_caching) caps.push('Caching');
    return caps;
  }, [model.supports_function_calling, model.supports_reasoning, model.supports_prompt_caching]);

  if (capabilities.length === 0) {
    return null;
  }

  return (
    <div
      css={{
        marginTop: theme.spacing.xs,
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.xs,
        flexWrap: 'wrap',
      }}
    >
      {capabilities.map((cap) => (
        <Tag key={cap} componentId={`mlflow.gateway.model-select.capability.${cap.toLowerCase()}`}>
          {cap}
        </Tag>
      ))}
    </div>
  );
});
