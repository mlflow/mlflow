import React, { useMemo, useState, useCallback, useEffect, useRef } from 'react';
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
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import { useEndpointsQuery } from '../../gateway/hooks/useEndpointsQuery';
import { CreateEndpointModal } from '../../gateway/components/endpoint-form';
import { getEndpointDisplayInfo } from '../../gateway/utils/gatewayUtils';
import type { Endpoint } from '../../gateway/types';
import { Link } from '../../common/utils/RoutingUtils';
import GatewayRoutes from '../../gateway/routes';

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
  /** Whether to show the "Create new endpoint" button. Defaults to true. */
  showCreateButton?: boolean;
  /** Whether to auto-select the first endpoint */
  autoSelectFirstEndpoint?: boolean;
  /** Called when the current endpoint name doesn't match any loaded endpoint (e.g., after a rename) */
  onEndpointNotFound?: () => void;
  /** Endpoint IDs to exclude from the list (e.g., the guarded endpoint itself) */
  excludeEndpointIds?: string[];
  /** Optional max width for the trigger; long endpoint labels truncate with ellipsis when this is set. */
  triggerMaxWidth?: number | string;
}

export const EndpointSelector: React.FC<EndpointSelectorProps> = ({
  currentEndpointName,
  onEndpointSelect,
  disabled = false,
  placeholder,
  componentIdPrefix = 'mlflow.endpoint-selector',
  onEndpointCreated,
  showCreateButton = true,
  autoSelectFirstEndpoint = false,
  onEndpointNotFound,
  excludeEndpointIds,
  triggerMaxWidth,
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
    return endpoints
      .filter((endpoint) => !excludeEndpointIds?.includes(endpoint.endpoint_id))
      .map((endpoint) => {
        const displayInfo = getEndpointDisplayInfo(endpoint);
        return {
          value: endpoint.name,
          label: endpoint.name,
          provider: displayInfo?.provider,
          modelName: displayInfo?.modelName,
        };
      });
  }, [endpoints, excludeEndpointIds]);

  const currentEndpoint = useMemo(() => {
    return endpointOptions.find((opt) => opt.value === currentEndpointName);
  }, [endpointOptions, currentEndpointName]);

  // When the endpoint name doesn't match any loaded endpoint (e.g., after a rename),
  // notify the parent so it can refetch scorer data with the resolved endpoint name.
  const hasTriedRefetch = useRef(false);

  useEffect(() => {
    hasTriedRefetch.current = false;
  }, [currentEndpointName]);

  useEffect(() => {
    if (currentEndpointName && !currentEndpoint && !isLoading && !error && !hasTriedRefetch.current) {
      hasTriedRefetch.current = true;
      onEndpointNotFound?.();
    }
  }, [currentEndpointName, currentEndpoint, isLoading, error, onEndpointNotFound]);

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
        componentId="mlflow.endpoint-selector.endpoints-error"
        type="error"
        message={error.message || 'Failed to load endpoints'}
      />
    );
  }

  return (
    <>
      <DialogCombobox
        componentId="mlflow.endpoint-selector.select"
        id={`${componentIdPrefix}.select`}
        value={currentEndpointName ? [currentEndpointName] : []}
      >
        <DialogComboboxTrigger
          withInlineLabel={false}
          allowClear={false}
          disabled={disabled}
          maxWidth={triggerMaxWidth}
          placeholder={placeholder || defaultPlaceholder}
          renderDisplayedValue={() => {
            // Endpoint not found in list - may have been deleted
            if (currentEndpointName && !currentEndpoint) {
              return (
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                  <span>{currentEndpointName}</span>
                  <Tooltip
                    componentId="mlflow.endpoint-selector.deleted-endpoint-tooltip"
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
            {endpointOptions.length === 0 ? (
              <div
                css={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: theme.spacing.sm,
                  // The popover content sets no horizontal padding on raw children and has a 320px min width,
                  // so pad/cap here to match the option items and keep the text from forcing the popover wider.
                  boxSizing: 'border-box',
                  maxWidth: 320,
                  padding: `${theme.spacing.sm}px ${theme.spacing.lg}px`,
                }}
              >
                <Typography.Text color="secondary">
                  <FormattedMessage
                    defaultMessage="No endpoints available. Set up an endpoint in the AI Gateway to get started."
                    description="Empty state message shown in the endpoint selector when no endpoints are configured"
                  />
                </Typography.Text>
                <Link
                  componentId="mlflow.endpoint-selector.no-endpoints-gateway-link"
                  to={GatewayRoutes.gatewayPageRoute}
                >
                  <FormattedMessage
                    defaultMessage="Go to AI Gateway"
                    description="Link in the endpoint selector empty state that navigates to the AI Gateway page"
                  />
                </Link>
              </div>
            ) : (
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
            )}
            {showCreateButton && (
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

      {showCreateButton && (
        <CreateEndpointModal
          open={isCreateModalOpen}
          onClose={handleCloseCreateModal}
          onSuccess={handleCreateEndpointSuccess}
        />
      )}
    </>
  );
};
