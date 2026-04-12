import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  FormUI,
  Tooltip,
  Typography,
  useDesignSystemTheme,
  TrashIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useState, useRef } from 'react';
import { useDrag, useDrop } from 'react-dnd';
import type { FallbackModel } from '../../hooks/useEditEndpointForm';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { ApiKeyConfigurator } from '../model-configuration/components/ApiKeyConfigurator';
import { useApiKeyConfiguration } from '../model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../model-configuration/types';
import { formatProviderName } from '../../utils/providerUtils';

const FALLBACK_MODEL_DRAG_TYPE = 'FALLBACK_MODEL';

interface DragItem {
  type: string;
  index: number;
}

interface FallbackModelItemProps {
  model: FallbackModel;
  index: number;
  onModelChange: (index: number, updates: Partial<FallbackModel>) => void;
  onRemove: (index: number) => void;
  onMove: (fromIndex: number, toIndex: number) => void;
  componentId: string;
}

export const FallbackModelItem = ({
  model,
  index,
  onModelChange,
  onRemove,
  onMove,
  componentId,
}: FallbackModelItemProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isExpanded, setIsExpanded] = useState(!model.provider && !model.modelName);
  const ref = useRef<HTMLDivElement>(null);

  const [{ isDragging }, drag, preview] = useDrag({
    type: FALLBACK_MODEL_DRAG_TYPE,
    item: { type: FALLBACK_MODEL_DRAG_TYPE, index },
    collect: (monitor) => ({
      isDragging: monitor.isDragging(),
    }),
  });

  const [{ isOver }, drop] = useDrop<DragItem, void, { isOver: boolean }>({
    accept: FALLBACK_MODEL_DRAG_TYPE,
    hover: (item: DragItem) => {
      if (!ref.current) {
        return;
      }
      const dragIndex = item.index;
      const hoverIndex = index;

      if (dragIndex === hoverIndex) {
        return;
      }

      onMove(dragIndex, hoverIndex);
      item.index = hoverIndex;
    },
    collect: (monitor) => ({
      isOver: monitor.isOver(),
    }),
  });

  preview(drop(ref));

  const { existingSecrets, isLoadingSecrets, authModes, defaultAuthMode, isLoadingProviderConfig } =
    useApiKeyConfiguration({
      provider: model.provider,
    });

  const apiKeyConfig: ApiKeyConfiguration = {
    mode: model.secretMode,
    existingSecretId: model.existingSecretId,
    newSecret: model.newSecret,
  };

  const handleApiKeyChange = (config: ApiKeyConfiguration) => {
    onModelChange(index, {
      secretMode: config.mode,
      existingSecretId: config.existingSecretId,
      newSecret: config.newSecret,
    });
  };

  const hasModelInfo = model.provider && model.modelName;

  return (
    <div
      ref={ref}
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        border: `2px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.backgroundPrimary,
        opacity: isDragging ? 0.5 : 1,
        borderColor: isOver ? theme.colors.actionPrimaryBackgroundDefault : theme.colors.border,
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div
          ref={drag}
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.sm,
            flex: 1,
            cursor: 'grab',
            '&:active': { cursor: 'grabbing' },
          }}
        >
          <FormUI.Label css={{ margin: 0, fontWeight: 'bold' }}>
            <FormattedMessage
              defaultMessage="Fallback Model {order}"
              description="Label for fallback model"
              values={{ order: index + 1 }}
            />
          </FormUI.Label>
          <Button
            componentId={`${componentId}.expand`}
            icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={() => setIsExpanded(!isExpanded)}
            size="small"
          />
          {!isExpanded && hasModelInfo && (
            <Typography.Text css={{ fontFamily: 'monospace', color: theme.colors.textSecondary }}>
              {formatProviderName(model.provider)} / {model.modelName}
            </Typography.Text>
          )}
        </div>
        <Tooltip
          componentId={`${componentId}.remove-tooltip`}
          content={intl.formatMessage({
            defaultMessage: 'Remove fallback model',
            description: 'Tooltip for remove fallback model button',
          })}
        >
          <Button componentId={`${componentId}.remove`} icon={<TrashIcon />} onClick={() => onRemove(index)} />
        </Tooltip>
      </div>

      {isExpanded && (
        <>
          <ProviderSelect
            value={model.provider}
            onChange={(provider) => {
              onModelChange(index, {
                provider,
                modelName: '',
                secretMode: 'new',
                existingSecretId: '',
                newSecret: {
                  name: '',
                  authMode: '',
                  secretFields: {},
                  configFields: {},
                },
              });
            }}
            componentId={`${componentId}.provider`}
          />

          <ModelSelect
            provider={model.provider}
            value={model.modelName}
            onChange={(modelName) => onModelChange(index, { modelName })}
            componentId={`${componentId}.model`}
          />

          <ApiKeyConfigurator
            value={apiKeyConfig}
            onChange={handleApiKeyChange}
            provider={model.provider}
            existingSecrets={existingSecrets}
            isLoadingSecrets={isLoadingSecrets}
            authModes={authModes}
            defaultAuthMode={defaultAuthMode}
            isLoadingProviderConfig={isLoadingProviderConfig}
            componentId={`${componentId}.api-key`}
          />
        </>
      )}
    </div>
  );
};
