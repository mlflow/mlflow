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
import { useState } from 'react';
import { GatewayInput } from '../common';
import type { TrafficSplitModel } from '../../hooks/useEditEndpointForm';
import { ProviderSelect } from '../create-endpoint/ProviderSelect';
import { ModelSelect } from '../create-endpoint/ModelSelect';
import { ApiKeyConfigurator } from '../model-configuration/components/ApiKeyConfigurator';
import { useApiKeyConfiguration } from '../model-configuration/hooks/useApiKeyConfiguration';
import type { ApiKeyConfiguration } from '../model-configuration/types';
import { formatProviderName } from '../../utils/providerUtils';

interface TrafficSplitModelItemProps {
  model: TrafficSplitModel;
  index: number;
  onModelChange: (index: number, updates: Partial<TrafficSplitModel>) => void;
  onWeightChange: (index: number, weight: number) => void;
  onRemove: (index: number) => void;
  componentIdPrefix: string;
}

export const TrafficSplitModelItem = ({
  model,
  index,
  onModelChange,
  onWeightChange,
  onRemove,
  componentIdPrefix,
}: TrafficSplitModelItemProps) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const [isExpanded, setIsExpanded] = useState(!model.provider && !model.modelName);

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
      css={{
        display: 'flex',
        flexDirection: 'column',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        border: `2px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        backgroundColor: theme.colors.white,
      }}
    >
      <div css={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm, flex: 1 }}>
          <Button
            componentId={`${componentIdPrefix}.expand.${index}`}
            icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
            onClick={() => setIsExpanded(!isExpanded)}
            size="small"
          />
          <FormUI.Label css={{ margin: 0, fontWeight: 'bold' }}>
            <FormattedMessage
              defaultMessage="Model {number}"
              description="Label for traffic split model"
              values={{ number: index + 1 }}
            />
          </FormUI.Label>
          {!isExpanded && hasModelInfo && (
            <Typography.Text css={{ fontFamily: 'monospace', color: theme.colors.textSecondary }}>
              {formatProviderName(model.provider)} / {model.modelName}
              <span css={{ marginLeft: theme.spacing.sm, color: theme.colors.actionTertiaryTextDefault }}>
                ({model.weight}%)
              </span>
            </Typography.Text>
          )}
        </div>
        <Tooltip
          componentId={`${componentIdPrefix}.remove-tooltip.${index}`}
          content={intl.formatMessage({
            defaultMessage: 'Remove model',
            description: 'Tooltip for remove traffic split model button',
          })}
        >
          <Button
            componentId={`${componentIdPrefix}.remove.${index}`}
            icon={<TrashIcon />}
            onClick={() => onRemove(index)}
          />
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
            componentIdPrefix={`${componentIdPrefix}.provider.${index}`}
          />

          <ModelSelect
            provider={model.provider}
            value={model.modelName}
            onChange={(modelName) => onModelChange(index, { modelName })}
            componentIdPrefix={`${componentIdPrefix}.model.${index}`}
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
            componentIdPrefix={`${componentIdPrefix}.api-key.${index}`}
          />

          <div css={{ width: 120 }}>
            <FormUI.Label htmlFor={`${componentIdPrefix}.weight.${index}`}>
              <FormattedMessage defaultMessage="Weight" description="Label for traffic split weight input" />
            </FormUI.Label>
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
              <GatewayInput
                componentId={`${componentIdPrefix}.weight.${index}`}
                type="number"
                min={0}
                max={100}
                value={model.weight}
                onChange={(e) => {
                  const parsed = parseInt(e.target.value, 10);
                  if (!Number.isNaN(parsed)) {
                    onWeightChange(index, parsed);
                  }
                }}
                css={{ width: '100%' }}
              />
              <span css={{ color: theme.colors.textSecondary }}>%</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
