import { Radio, useDesignSystemTheme, FormUI, Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ModelDefinitionSelector } from './ModelDefinitionSelector';
import { useModelDefinitionsQuery } from '../../hooks/useModelDefinitionsQuery';

export type ModelDefinitionMode = 'existing' | 'new';

interface ModelDefinitionConfigSectionProps {
  mode: ModelDefinitionMode;
  onModeChange: (mode: ModelDefinitionMode) => void;
  selectedModelDefinitionId?: string;
  onModelDefinitionSelect: (modelDefinitionId: string) => void;
  error?: string;
  componentIdPrefix?: string;
  children?: React.ReactNode;
}

export const ModelDefinitionConfigSection = ({
  mode,
  onModeChange,
  selectedModelDefinitionId,
  onModelDefinitionSelect,
  error,
  componentIdPrefix = 'mlflow.gateway.model-definition-config',
  children,
}: ModelDefinitionConfigSectionProps) => {
  const { theme } = useDesignSystemTheme();
  const { data: modelDefinitions } = useModelDefinitionsQuery();

  const hasExistingModelDefinitions = modelDefinitions && modelDefinitions.length > 0;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <FormUI.Label>
          <FormattedMessage defaultMessage="Model configuration" description="Label for model configuration section" />
        </FormUI.Label>
        <Radio.Group
          componentId={`${componentIdPrefix}.mode`}
          name="modelDefinitionMode"
          value={mode}
          onChange={(e) => onModeChange(e.target.value as ModelDefinitionMode)}
          layout="horizontal"
        >
          <Radio value="new">
            <FormattedMessage defaultMessage="Configure new model" description="Option to configure new model" />
          </Radio>
          <Radio value="existing" disabled={!hasExistingModelDefinitions}>
            <FormattedMessage
              defaultMessage="Use existing model definition"
              description="Option to use existing model definition"
            />
          </Radio>
        </Radio.Group>
        {!hasExistingModelDefinitions && mode === 'new' && (
          <Typography.Text
            color="secondary"
            css={{ display: 'block', marginTop: theme.spacing.xs, fontSize: theme.typography.fontSizeSm }}
          >
            <FormattedMessage
              defaultMessage="No existing model definitions. Create a new one below."
              description="Message when no existing model definitions"
            />
          </Typography.Text>
        )}
      </div>

      {mode === 'existing' ? (
        <ModelDefinitionSelector
          value={selectedModelDefinitionId ?? ''}
          onChange={onModelDefinitionSelect}
          error={error}
        />
      ) : (
        children
      )}
    </div>
  );
};
