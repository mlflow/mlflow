import { useCallback } from 'react';
import { Button, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import type { FallbackModel } from '../../hooks/useEditEndpointForm';
import { FallbackModelItem } from './FallbackModelItem';

export interface FallbackModelsConfiguratorProps {
  value: FallbackModel[];
  onChange: (value: FallbackModel[]) => void;
  componentId?: string;
}

export const FallbackModelsConfigurator = ({
  value,
  onChange,
  componentId = 'mlflow.gateway.fallback',
}: FallbackModelsConfiguratorProps) => {
  const { theme } = useDesignSystemTheme();

  const handleAddModel = useCallback(() => {
    const nextOrder = value.length > 0 ? Math.max(...value.map((m) => m.fallbackOrder)) + 1 : 1;

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
        fallbackOrder: nextOrder,
      },
    ]);
  }, [value, onChange]);

  const handleRemoveModel = useCallback(
    (index: number) => {
      const newValue = value.filter((_, i) => i !== index);
      const reorderedValue = newValue.map((model, idx) => ({
        ...model,
        fallbackOrder: idx + 1,
      }));
      onChange(reorderedValue);
    },
    [value, onChange],
  );

  const handleModelChange = useCallback(
    (index: number, updates: Partial<FallbackModel>) => {
      const newValue = [...value];
      newValue[index] = { ...newValue[index], ...updates };
      onChange(newValue);
    },
    [value, onChange],
  );

  const handleMoveModel = useCallback(
    (fromIndex: number, toIndex: number) => {
      const newValue = [...value];
      const [movedItem] = newValue.splice(fromIndex, 1);
      newValue.splice(toIndex, 0, movedItem);

      const updatedValue = newValue.map((model, idx) => ({
        ...model,
        fallbackOrder: idx + 1,
      }));

      onChange(updatedValue);
    },
    [value, onChange],
  );

  const connectorLine = (
    <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div css={{ width: 2, height: theme.spacing.md, backgroundColor: theme.colors.border }} />
      <Typography.Text
        color="secondary"
        css={{ padding: `${theme.spacing.xs}px ${theme.spacing.sm}px`, fontSize: theme.typography.fontSizeSm }}
      >
        <FormattedMessage
          defaultMessage="Fallback"
          description="Gateway > Endpoint details > Label on connector between fallback model items"
        />
      </Typography.Text>
      <div css={{ width: 2, height: theme.spacing.md, backgroundColor: theme.colors.border }} />
    </div>
  );

  return (
    <DndProvider backend={HTML5Backend}>
      <div css={{ display: 'flex', flexDirection: 'column' }}>
        {value.map((model, index) => (
          <div key={index}>
            {index > 0 && connectorLine}
            <FallbackModelItem
              model={model}
              index={index}
              onModelChange={handleModelChange}
              onRemove={handleRemoveModel}
              onMove={handleMoveModel}
              componentId={componentId}
            />
          </div>
        ))}

        {/* Connector line leading to Add fallback button (only when there are fallback models, otherwise the parent connector handles it) */}
        {value.length > 0 && (
          <div css={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div css={{ width: 2, height: theme.spacing.md, backgroundColor: theme.colors.border }} />
          </div>
        )}

        <Button componentId={`${componentId}.add`} onClick={handleAddModel} css={{ alignSelf: 'center' }}>
          <FormattedMessage defaultMessage="Add fallback" description="Button to add fallback model" />
        </Button>
      </div>
    </DndProvider>
  );
};
