import { useCallback } from 'react';
import { Button, useDesignSystemTheme, PlusIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import type { TrafficSplitModel } from '../../hooks/useEditEndpointForm';
import { TrafficSplitModelItem } from './TrafficSplitModelItem';

export interface TrafficSplitConfiguratorProps {
  value: TrafficSplitModel[];
  onChange: (value: TrafficSplitModel[]) => void;
  componentIdPrefix?: string;
}

export const TrafficSplitConfigurator = ({
  value,
  onChange,
  componentIdPrefix = 'mlflow.gateway.traffic-split',
}: TrafficSplitConfiguratorProps) => {
  const { theme } = useDesignSystemTheme();

  const handleAddModel = useCallback(() => {
    onChange([
      ...value,
      {
        modelDefinitionName: '',
        provider: '',
        modelName: '',
        secretMode: 'new' as const,
        existingSecretId: '',
        newSecret: {
          name: '',
          authMode: '',
          secretFields: {},
          configFields: {},
        },
        weight: 0,
      },
    ]);
  }, [value, onChange]);

  const handleRemoveModel = useCallback(
    (index: number) => {
      onChange(value.filter((_, i) => i !== index));
    },
    [value, onChange],
  );

  const handleModelChange = useCallback(
    (index: number, updates: Partial<TrafficSplitModel>) => {
      const newValue = [...value];
      newValue[index] = { ...newValue[index], ...updates };
      onChange(newValue);
    },
    [value, onChange],
  );

  const handleWeightChange = useCallback(
    (index: number, weight: number) => {
      const newValue = [...value];
      newValue[index] = { ...newValue[index], weight };
      onChange(newValue);
    },
    [value, onChange],
  );

  const totalWeight = value.reduce((sum, m) => sum + m.weight, 0);
  const isValidTotal = totalWeight === 100;

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {value.map((model, index) => (
        <TrafficSplitModelItem
          key={index}
          model={model}
          index={index}
          onModelChange={handleModelChange}
          onWeightChange={handleWeightChange}
          onRemove={handleRemoveModel}
          componentIdPrefix={componentIdPrefix}
        />
      ))}

      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Button componentId={`${componentIdPrefix}.add`} icon={<PlusIcon />} onClick={handleAddModel}>
          <FormattedMessage
            defaultMessage="Add model for traffic split"
            description="Button to add model for traffic split"
          />
        </Button>

        {value.length > 0 && (
          <div
            css={{
              display: 'flex',
              alignItems: 'center',
              gap: theme.spacing.xs,
              color: isValidTotal ? theme.colors.textSecondary : theme.colors.textValidationDanger,
              fontSize: theme.typography.fontSizeSm,
            }}
          >
            <FormattedMessage
              defaultMessage="Total: {total}%"
              description="Total weight display"
              values={{ total: totalWeight }}
            />
            {!isValidTotal && (
              <FormattedMessage defaultMessage="(must equal 100%)" description="Weight validation message" />
            )}
          </div>
        )}
      </div>
    </div>
  );
};
