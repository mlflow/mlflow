import { useCallback } from 'react';
import { Button, useDesignSystemTheme, PlusIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import type { FallbackModel } from '../../hooks/useEditEndpointForm';
import { FallbackModelItem } from './FallbackModelItem';

export interface FallbackModelsConfiguratorProps {
  value: FallbackModel[];
  onChange: (value: FallbackModel[]) => void;
  componentIdPrefix?: string;
}

export const FallbackModelsConfigurator = ({
  value,
  onChange,
  componentIdPrefix = 'mlflow.gateway.fallback',
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

  return (
    <DndProvider backend={HTML5Backend}>
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        {value.map((model, index) => (
          <FallbackModelItem
            key={index}
            model={model}
            index={index}
            onModelChange={handleModelChange}
            onRemove={handleRemoveModel}
            onMove={handleMoveModel}
            componentIdPrefix={componentIdPrefix}
          />
        ))}

        <Button componentId={`${componentIdPrefix}.add`} icon={<PlusIcon />} onClick={handleAddModel}>
          <FormattedMessage defaultMessage="Add fallback" description="Button to add fallback model" />
        </Button>
      </div>
    </DndProvider>
  );
};
