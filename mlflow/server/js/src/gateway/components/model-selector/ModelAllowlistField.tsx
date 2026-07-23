import { useState, useCallback } from 'react';
import { Button, PlusIcon, Tag, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ModelSelectorModal } from './ModelSelectorModal';
import type { ProviderModel } from '../../types';

interface ModelAllowlistFieldProps {
  /** Provider the allowlisted models must belong to. The add-model action is disabled until set. */
  provider: string;
  value: ProviderModel[];
  onChange: (models: ProviderModel[]) => void;
  /** Prefix used to derive stable componentIds for telemetry. */
  componentId: string;
}

export const ModelAllowlistField = ({ provider, value, onChange, componentId }: ModelAllowlistFieldProps) => {
  const { theme } = useDesignSystemTheme();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleSelect = useCallback(
    (model: ProviderModel) => {
      if (!value.some((m) => m.model === model.model)) {
        onChange([...value, model]);
      }
      setIsModalOpen(false);
    },
    [value, onChange],
  );

  const handleRemove = useCallback(
    (modelName: string) => {
      onChange(value.filter((m) => m.model !== modelName));
    },
    [value, onChange],
  );

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      {value.length > 0 && (
        <div css={{ display: 'flex', flexWrap: 'wrap', gap: theme.spacing.xs }}>
          {value.map((model) => (
            <Tag
              key={model.model}
              componentId={`${componentId}.chip`}
              closable
              onClose={() => handleRemove(model.model)}
              css={{ margin: 0 }}
            >
              {model.model}
            </Tag>
          ))}
        </div>
      )}
      <div>
        <Button
          componentId={`${componentId}.add-model`}
          icon={<PlusIcon />}
          disabled={!provider}
          onClick={() => setIsModalOpen(true)}
        >
          <FormattedMessage
            defaultMessage="Add model"
            description="Button to add a model to a connection's allowlist"
          />
        </Button>
        {!provider && (
          <Typography.Text color="secondary" size="sm" css={{ display: 'block', marginTop: theme.spacing.xs }}>
            <FormattedMessage
              defaultMessage="Select a provider first to choose models."
              description="Hint shown when no provider is selected yet for the model allowlist"
            />
          </Typography.Text>
        )}
      </div>
      <ModelSelectorModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSelect={handleSelect}
        provider={provider}
      />
    </div>
  );
};
