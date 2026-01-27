import React, { useMemo, useState, useCallback, useEffect } from 'react';
import {
  useDesignSystemTheme,
  Typography,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxAddButton,
  DialogComboboxFooter,
  DialogComboboxOptionList,
  DialogComboboxOptionListSelectItem,
  DialogComboboxHintRow,
  DialogComboboxTrigger,
  Tooltip,
  Spinner,
  InfoSmallIcon,
  Alert,
  ButtonSize,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useEndpointsQuery } from '../../gateway/hooks/useEndpointsQuery';
import { CreateEndpointModal } from '../../gateway/components/endpoint-form';
import { getEndpointDisplayInfo } from '../../gateway/utils/gatewayUtils';
import type { Endpoint } from '../../gateway/types';

interface EndpointOption {
  value: string;
  label: string;
  provider?: string;
  modelName?: string;
}

export interface EndpointSelectorProps {
  /** Current selected endpoint name */
  currentEndpointName?: string;
  /** Called when user selects an endpoint */
  onEndpointSelect: (endpointName: string) => void;
  /** Whether the selector is disabled/read-only */
  disabled?: boolean;
  /** Optional placeholder text */
  placeholder?: string;
  /** Component ID prefix for tracking */
  componentIdPrefix?: string;
  /** Called when a new endpoint is created */
  onEndpointCreated?: (endpoint: Endpoint) => void;
  /** Whether to hide the create new endpoint button */
  hideCreateNewEndpointButton?: boolean;
  /** Size of the trigger */
  triggerSize?: ButtonSize;
  /** Whether to auto-select the first endpoint */
  autoSelectFirstEndpoint?: boolean;
}

export const EndpointSelector: React.FC<EndpointSelectorProps> = ({
  currentEndpointName,
  onEndpointSelect,
  disabled = false,
  placeholder,
  componentIdPrefix = 'mlflow.endpoint-selector',
  onEndpointCreated,
  hideCreateNewEndpointButton = false,
  triggerSize,
  autoSelectFirstEndpoint = false,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const { data: endpoints, isLoading, error, refetch } = useEndpointsQuery();

  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  useEffect(() => {
    if (autoSelectFirstEndpoint && endpoints && endpoints.length > 0 && !currentEndpointName) {
      onEndpointSelect(endpoints[0].name);
    }
  }, [autoSelectFirstEndpoint, endpoints, onEndpointSelect, currentEndpointName]);

  const handleOpenCreateModal = useCallback(() => {
    setIsCreateModalOpen(true);
  }, []);

  const handleCloseCreateModal = useCallback(() => {
    setIsCreateModalOpen(false);
  }, []);

  const handleCreateEndpointSuccess = useCallback(
    async (endpoint: Endpoint) => {
      await refetch();
      onEndpointSelect(endpoint.name);
      onEndpointCreated?.(endpoint);
      setIsCreateModalOpen(false);
    },
    [refetch, onEndpointSelect, onEndpointCreated],
  );

  // Build endpoint options for the dropdown
  const endpointOptions: EndpointOption[] = useMemo(() => {
    return endpoints.map((endpoint) => {
      const displayInfo = getEndpointDisplayInfo(endpoint);
      return {
        value: endpoint.name,
        label: endpoint.name,
        provider: displayInfo?.provider,
        modelName: displayInfo?.modelName,
      };
    });
  }, [endpoints]);

  const currentEndpoint = useMemo(() => {
    return endpointOptions.find((opt) => opt.value === currentEndpointName);
  }, [endpointOptions, currentEndpointName]);

  const defaultPlaceholder = intl.formatMessage({
    defaultMessage: 'Select an endpoint',
    description: 'Placeholder for endpoint selection dropdown',
  });

  if (isLoading) {
    return (
      <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
        <Spinner size="small" />
        <Typography.Text color="secondary">
          <FormattedMessage defaultMessage="Loading endpoints..." description="Loading endpoints message" />
        </Typography.Text>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        componentId={`${componentIdPrefix}.endpoints-error`}
        type="error"
        message={error.message || 'Failed to load endpoints'}
      />
    );
  }

  return (
    <>
      <DialogCombobox
        componentId={`${componentIdPrefix}.select`}
        id={`${componentIdPrefix}.select`}
        value={currentEndpointName ? [currentEndpointName] : []}
      >
        <DialogComboboxTrigger
          triggerSize={triggerSize}
          withInlineLabel={false}
          allowClear={false}
          disabled={disabled}
          placeholder={placeholder || defaultPlaceholder}
          renderDisplayedValue={() => {
            // Endpoint not found in list - may have been deleted
            if (currentEndpointName && !currentEndpoint) {
              return (
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <span>{currentEndpointName}</span>
                  <Tooltip
                    componentId={`${componentIdPrefix}.deleted-endpoint-tooltip`}
                    content={intl.formatMessage({
                      defaultMessage: 'This endpoint may have been deleted',
                      description: 'Tooltip for deleted endpoint',
                    })}
                  >
                    <InfoSmallIcon css={{ color: theme.colors.textValidationWarning }} />
                  </Tooltip>
                </div>
              );
            }

            if (!currentEndpoint) {
              return null;
            }

            // Normal endpoint display
            return (
              <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                <span>{currentEndpoint.label}</span>
                {currentEndpoint.provider && currentEndpoint.modelName && (
                  <Typography.Text color="secondary" size="sm">
                    ({currentEndpoint.provider} / {currentEndpoint.modelName})
                  </Typography.Text>
                )}
              </div>
            );
          }}
        />
        {!disabled && (
          <DialogComboboxContent maxHeight={350}>
            <DialogComboboxOptionList>
              {endpointOptions.map((option) => (
                <DialogComboboxOptionListSelectItem
                  key={option.value}
                  value={option.value}
                  onChange={() => onEndpointSelect(option.value)}
                  checked={currentEndpointName === option.value}
                >
                  {option.label}
                  {option.provider && option.modelName && (
                    <DialogComboboxHintRow>
                      {option.provider} / {option.modelName}
                    </DialogComboboxHintRow>
                  )}
                </DialogComboboxOptionListSelectItem>
              ))}
            </DialogComboboxOptionList>
            {!hideCreateNewEndpointButton && (
              <DialogComboboxFooter>
                <DialogComboboxAddButton onClick={handleOpenCreateModal}>
                  <FormattedMessage
                    defaultMessage="Create new endpoint"
                    description="Button text to create a new endpoint"
                  />
                </DialogComboboxAddButton>
              </DialogComboboxFooter>
            )}
          </DialogComboboxContent>
        )}
      </DialogCombobox>

      <CreateEndpointModal
        open={isCreateModalOpen}
        onClose={handleCloseCreateModal}
        onSuccess={handleCreateEndpointSuccess}
      />
    </>
  );
};
