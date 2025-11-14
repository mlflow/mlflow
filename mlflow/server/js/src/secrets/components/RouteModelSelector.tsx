import {
  Button,
  FormUI,
  Input,
  RefreshIcon,
  SimpleSelect,
  SimpleSelectOption,
  SimpleSelectOptionGroup,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { useMemo } from 'react';
import type { Model, Provider } from './routeConstants';
import { groupModelsByFamily } from './routeUtils';

interface RouteModelSelectorProps {
  provider?: Provider;
  availableModels: Model[];
  selectedModelFromList: string | null;
  onSelectModelFromList: (modelId: string) => void;
  modelName: string;
  onChangeModelName: (name: string) => void;
  showCustomModelInput: boolean;
  onShowCustomModelInput: (show: boolean) => void;
  onFetchModels?: () => void;
  isFetchingModels?: boolean;
  error?: string;
  componentIdPrefix?: string;
}

export const RouteModelSelector = ({
  provider,
  availableModels,
  selectedModelFromList,
  onSelectModelFromList,
  modelName,
  onChangeModelName,
  showCustomModelInput,
  onShowCustomModelInput,
  onFetchModels,
  isFetchingModels = false,
  error,
  componentIdPrefix = 'mlflow.routes.model_selector',
}: RouteModelSelectorProps) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  const groupedModels = useMemo(() => groupModelsByFamily(availableModels), [availableModels]);

  return (
    <div>
      {provider?.supportsModelFetch && onFetchModels && (
        <div css={{ marginBottom: theme.spacing.md, display: 'flex', justifyContent: 'flex-end' }}>
          <Button
            componentId={`${componentIdPrefix}.fetch_models`}
            onClick={onFetchModels}
            loading={isFetchingModels}
            icon={<RefreshIcon />}
            size="small"
            type="tertiary"
          >
            <FormattedMessage defaultMessage="Fetch models from provider" description="Fetch models button" />
          </Button>
        </div>
      )}

      {availableModels.length > 0 ? (
        <>
          <FormUI.Label htmlFor="model-list-select">
            <FormattedMessage defaultMessage="Available Models" description="Available models label" />
          </FormUI.Label>
          <SimpleSelect
            componentId={`${componentIdPrefix}.model_list`}
            id="model-list-select"
            label=""
            value={selectedModelFromList || ''}
            onChange={(e) => {
              onSelectModelFromList(e.target.value);
              onShowCustomModelInput(false);
            }}
            css={{ width: '100%' }}
            contentProps={{ style: { maxHeight: '500px', minWidth: '400px', overflow: 'auto' } }}
            placeholder={intl.formatMessage({
              defaultMessage: 'Click to choose a model...',
              description: 'Model select placeholder',
            })}
          >
            {groupedModels.map((group) => (
              <SimpleSelectOptionGroup key={group.groupName} label={group.groupName}>
                {group.models.map((model) => (
                  <SimpleSelectOption key={model.id} value={model.id}>
                    {model.id}
                  </SimpleSelectOption>
                ))}
              </SimpleSelectOptionGroup>
            ))}
          </SimpleSelect>
          {!showCustomModelInput && !selectedModelFromList && (
            <FormUI.Hint css={{ marginTop: theme.spacing.sm, display: 'flex', alignItems: 'center', gap: 4 }}>
              <span>
                <FormattedMessage defaultMessage="Or " description="Custom model hint prefix" />
              </span>
              <Button
                componentId={`${componentIdPrefix}.show_custom_model`}
                type="link"
                size="small"
                onClick={() => onShowCustomModelInput(true)}
                css={{ padding: 0, height: 'auto', lineHeight: 'inherit' }}
              >
                <FormattedMessage defaultMessage="enter a custom model name" description="Custom model link" />
              </Button>
            </FormUI.Hint>
          )}
        </>
      ) : null}

      {(showCustomModelInput || availableModels.length === 0) && (
        <div css={{ marginTop: availableModels.length > 0 ? theme.spacing.md : 0 }}>
          <FormUI.Label htmlFor="model-name-input">
            <FormattedMessage defaultMessage="Model Name" description="Model name label" />
          </FormUI.Label>
          <Input
            componentId={`${componentIdPrefix}.model_name`}
            id="model-name-input"
            autoComplete="off"
            placeholder={intl.formatMessage({
              defaultMessage: 'e.g., gpt-4o, claude-sonnet-4-5-20250929',
              description: 'Model name placeholder',
            })}
            value={modelName}
            onChange={(e) => {
              onChangeModelName(e.target.value);
              onSelectModelFromList('');
            }}
            validationState={error ? 'error' : undefined}
          />
          {error && <FormUI.Message type="error" message={error} />}
        </div>
      )}
    </div>
  );
};
