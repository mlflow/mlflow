import { SimpleSelect, SimpleSelectOption, FormUI, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { useModelDefinitionsQuery } from '../../hooks/useModelDefinitionsQuery';
import { formatProviderName } from '../../utils/providerUtils';

interface ModelDefinitionSelectorProps {
  value: string;
  onChange: (modelDefinitionId: string) => void;
  provider?: string;
  disabled?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

export const ModelDefinitionSelector = ({
  value,
  onChange,
  provider,
  disabled,
  error,
  componentIdPrefix = 'mlflow.gateway.model-definition',
}: ModelDefinitionSelectorProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: modelDefinitions, isLoading } = useModelDefinitionsQuery();

  const filteredModelDefinitions = provider
    ? modelDefinitions?.filter((md) => md.provider === provider)
    : modelDefinitions;

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <FormattedMessage
          defaultMessage="Loading model definitions..."
          description="Loading message for model definitions"
        />
      </div>
    );
  }

  const hasOptions = filteredModelDefinitions && filteredModelDefinitions.length > 0;
  const placeholder = hasOptions
    ? 'Select a model definition'
    : provider
      ? `No existing model definitions for ${formatProviderName(provider)}`
      : 'No model definitions available';

  const selectId = `${componentIdPrefix}.select`;

  return (
    <div css={{ width: '100%' }}>
      <FormUI.Label htmlFor={selectId}>
        <FormattedMessage defaultMessage="Model definition" description="Label for model definition selector" />
      </FormUI.Label>
      <SimpleSelect
        id={selectId}
        componentId={selectId}
        value={value}
        onChange={({ target }) => onChange(target.value)}
        disabled={disabled || !hasOptions}
        placeholder={placeholder}
        validationState={error ? 'error' : undefined}
        css={{ width: '100%' }}
        contentProps={{
          matchTriggerWidth: true,
          maxHeight: 300,
        }}
      >
        {filteredModelDefinitions?.map((modelDef) => (
          <SimpleSelectOption key={modelDef.model_definition_id} value={modelDef.model_definition_id}>
            <div css={{ display: 'flex', flexDirection: 'column' }}>
              <span>{modelDef.name}</span>
              <span css={{ fontSize: theme.typography.fontSizeSm, color: theme.colors.textSecondary }}>
                {modelDef.model_name}
              </span>
            </div>
          </SimpleSelectOption>
        ))}
      </SimpleSelect>
      {error && <FormUI.Message type="error" message={error} />}
    </div>
  );
};
