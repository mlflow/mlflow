import { useState, useRef, useMemo, useCallback, memo } from 'react';
import { Input, useDesignSystemTheme, FormUI, Tag, ModelsIcon } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { ModelSelectorModal } from '../model-selector/ModelSelectorModal';
import { useModelsQuery } from '../../hooks/useModelsQuery';
import type { ProviderModel } from '../../types';
import { getModelCapabilities } from '../../utils/getModelCapabilities';

interface ModelSelectProps {
  provider: string;
  value: string;
  onChange: (model: string) => void;
  disabled?: boolean;
  error?: string;
  /** Component ID for telemetry (default: 'mlflow.gateway.model-select') */
  componentId?: string;
  /** Custom label for the select field. If not provided, defaults to "Model" */
  label?: React.ReactNode;
  /** If true, hides the model capabilities tags (Tools, Reasoning, Caching) */
  hideCapabilities?: boolean;
}

export const ModelSelect = ({
  provider,
  value,
  onChange,
  disabled,
  error,
  componentId = 'mlflow.gateway.model-select',
  label,
  hideCapabilities,
}: ModelSelectProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const domId = useRef(`model-select-${Math.random().toString(36).slice(2, 9)}`).current;
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
      <FormUI.Label htmlFor={domId}>
        {label ?? <FormattedMessage defaultMessage="Model" description="Label for model select field" />}
      </FormUI.Label>
      <Input
        id={domId}
        componentId={componentId}
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
      {selectedModel && !hideCapabilities && <ModelCapabilities model={selectedModel} />}
      <ModelSelectorModal
        isOpen={isModalOpen}
        onClose={handleClose}
        onSelect={handleSelect}
        provider={provider}
        initialValue={value}
      />
    </div>
  );
};

const ModelCapabilities = memo(function ModelCapabilities({ model }: { model: ProviderModel }) {
  const { theme } = useDesignSystemTheme();

  const capabilities = getModelCapabilities(model);

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
        <Tag key={cap} componentId="mlflow.gateway.model-select.capability">
          {cap}
        </Tag>
      ))}
    </div>
  );
});
