import React, { useState, useCallback, useEffect, useImperativeHandle, forwardRef, useMemo } from 'react';
import {
  useDesignSystemTheme,
  Button,
  Typography,
  Tooltip,
  DialogCombobox,
  DialogComboboxContent,
  DialogComboboxTrigger,
  DialogComboboxOptionList,
  DialogComboboxOptionListSearch,
  DialogComboboxOptionListSelectItem,
  DialogComboboxHintRow,
  DialogComboboxSeparator,
  Spinner,
  InfoSmallIcon,
  PlusIcon,
} from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { CreateEndpointModal } from '../../../../../gateway/components/endpoint-form';
import { ManualModelConfigModal } from './ManualModelConfigModal';
import type { ApiKeyConfiguration } from '../../../../../gateway/components/model-configuration/types';
import { useEndpointsQuery } from '../../../../../gateway/hooks/useEndpointsQuery';
import { useAllowlistedModelPairs } from '../../../../../gateway/hooks/useAllowlistedModelPairs';
import { getEndpointDisplayInfo } from '../../../../../gateway/utils/gatewayUtils';
import type { Endpoint } from '../../../../../gateway/types';
import { useNavigate } from '../../../../../common/utils/RoutingUtils';
import Routes from '../../../../routes';
import { SETTINGS_SECTION_LLM_CONNECTIONS } from '../../../../../settings/settingsSectionConstants';

/** Deep-link to the dedicated LLM Connections settings tab. */
const CONNECTIONS_ROUTE = Routes.getSettingsSectionRoute(SETTINGS_SECTION_LLM_CONNECTIONS);

type ModelConfigMode = 'endpoint' | 'direct';

const CONFIGURE_DIRECTLY_VALUE = '__configure_directly__';
const CREATE_ENDPOINT_VALUE = '__create_endpoint__';

export interface ModelSelectionValues {
  mode: ModelConfigMode;
  endpointName?: string;
  provider: string;
  model: string;
  apiKeyConfig: ApiKeyConfiguration;
  saveKey: boolean;
}

export interface GenAIModelSelectionRef {
  getValues: () => ModelSelectionValues;
  isValid: boolean;
  reset: () => void;
}

interface GenAIModelSelectionProps {
  onValidityChange: (isValid: boolean) => void;
  readOnly?: boolean;
  initialValues?: Partial<ModelSelectionValues>;
  /** Whether to show the "Configure model directly" option in the endpoint dropdown. Defaults to false. */
  showConfigureDirectly?: boolean;
  /** Whether to show the "Create Gateway endpoint" option in the endpoint dropdown. Defaults to false. */
  showCreateEndpoint?: boolean;
  /** Called whenever the selection changes (mode, endpoint, provider, model, etc.). */
  onSelectionChange?: (values: ModelSelectionValues) => void;
  /** Telemetry component ID prefix used for all child elements. */
  componentId: string;
  /** Subtitle shown below the "Select Model" heading. */
  description: React.ReactNode;
}

export const GenAIModelSelection = forwardRef<GenAIModelSelectionRef, GenAIModelSelectionProps>(
  function GenAIModelSelection(
    {
      onValidityChange,
      readOnly = false,
      initialValues,
      showConfigureDirectly = false,
      showCreateEndpoint = false,
      onSelectionChange,
      componentId,
      description,
    },
    ref,
  ) {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const navigate = useNavigate();

    // Fetch available endpoints
    const { data: endpoints, isLoading: isLoadingEndpoints, refetch: refetchEndpoints } = useEndpointsQuery();

    // Flat list of allowlisted "provider · model" pairs across all connections (direct mode).
    const { pairs: allowlistedPairs, isLoading: isLoadingPairs } = useAllowlistedModelPairs();
    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const hasEndpoints = endpoints.length > 0;

    // Manual entry escape hatch: when set, these values override the allowlisted-pair selection.
    const [manualValues, setManualValues] = useState<ModelSelectionValues | null>(null);
    const [isManualModalOpen, setIsManualModalOpen] = useState(false);

    // Track mode and selected endpoint - default to 'endpoint' mode if there are endpoints.
    // If initialValues provides a mode, use it directly and skip the endpoint-loading effect.
    const [mode, setMode] = useState<ModelConfigMode>(
      initialValues?.mode ?? (initialValues?.endpointName != null ? 'endpoint' : 'direct'),
    );
    const [selectedEndpointName, setSelectedEndpointName] = useState<string | undefined>(initialValues?.endpointName);
    const [hasInitializedMode, setHasInitializedMode] = useState(
      initialValues?.mode != null || initialValues?.endpointName != null,
    );

    // Set initial mode based on whether endpoints are available, and auto-select first endpoint
    useEffect(() => {
      if (!isLoadingEndpoints && !hasInitializedMode) {
        setMode(hasEndpoints ? 'endpoint' : 'direct');
        if (hasEndpoints && endpoints.length > 0) {
          setSelectedEndpointName(endpoints[0].name);
        }
        setHasInitializedMode(true);
      }
    }, [isLoadingEndpoints, hasEndpoints, hasInitializedMode, endpoints]);

    // Direct mode: the user picks a single allowlisted "provider · model" pair. The pair fully
    // determines provider + model + secret_id, so no inline provider/model/key entry is needed.
    // A pair is keyed by `${secretId}::${provider}::${model}`.
    const pairKey = useCallback(
      (p: { secretId: string; provider: string; model: string }) => `${p.secretId}::${p.provider}::${p.model}`,
      [],
    );
    const [selectedPairKey, setSelectedPairKey] = useState<string | undefined>(
      initialValues?.provider && initialValues?.model && initialValues?.apiKeyConfig?.existingSecretId
        ? `${initialValues.apiKeyConfig.existingSecretId}::${initialValues.provider}::${initialValues.model}`
        : undefined,
    );

    const selectedPair = useMemo(
      () => allowlistedPairs.find((p) => pairKey(p) === selectedPairKey),
      [allowlistedPairs, selectedPairKey, pairKey],
    );

    // Free-text search over the allowlisted "provider · model" pairs.
    const [pairSearch, setPairSearch] = useState('');
    const filteredPairs = useMemo(() => {
      const query = pairSearch.trim().toLowerCase();
      if (!query) return allowlistedPairs;
      return allowlistedPairs.filter((p) =>
        [p.label, p.provider, p.model, p.secretName].some((field) => field?.toLowerCase().includes(query)),
      );
    }, [allowlistedPairs, pairSearch]);

    // Auto-select the first available pair once loaded, if nothing is selected yet.
    useEffect(() => {
      if (!isLoadingPairs && !selectedPairKey && allowlistedPairs.length > 0) {
        setSelectedPairKey(pairKey(allowlistedPairs[0]));
      }
    }, [isLoadingPairs, selectedPairKey, allowlistedPairs, pairKey]);

    // Derive the ModelSelectionValues fields for direct mode from the selected pair. We reuse the
    // "existing secret" apiKeyConfig shape so IssueDetectionModal.handleSubmit passes secret_id
    // straight through without creating a new secret.
    const provider = selectedPair?.provider ?? '';
    const model = selectedPair?.model ?? '';
    const apiKeyConfig = useMemo<ApiKeyConfiguration>(
      () => ({
        mode: 'existing',
        existingSecretId: selectedPair?.secretId ?? '',
        newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
      }),
      [selectedPair],
    );
    // Pairs reference already-saved connections, so no secret needs to be (re)saved on submit.
    const saveKey = false;

    // Build endpoint options for the dropdown
    const endpointOptions = useMemo(() => {
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

    // Get display value for the dropdown
    const dropdownDisplayValue = useMemo(() => {
      if (hasInitializedMode && mode === 'direct' && showConfigureDirectly) {
        return intl.formatMessage({
          defaultMessage: 'Configure model directly',
          description: 'Option to configure model directly instead of using an endpoint',
        });
      }
      if (mode === 'endpoint' && selectedEndpointName) {
        const selectedOption = endpointOptions.find((opt) => opt.value === selectedEndpointName);
        return selectedOption?.label || '';
      }
      // No endpoint selected yet - return empty to show placeholder
      return '';
    }, [mode, selectedEndpointName, endpointOptions, intl, hasInitializedMode, showConfigureDirectly]);

    // Handle selection from dropdown
    const handleDropdownSelect = useCallback((value: string) => {
      if (value === CONFIGURE_DIRECTLY_VALUE) {
        setMode('direct');
        setSelectedEndpointName(undefined);
      } else {
        setMode('endpoint');
        setSelectedEndpointName(value);
      }
    }, []);

    const handleCreateEndpointSuccess = useCallback(
      async (endpoint: Endpoint) => {
        await refetchEndpoints();
        setMode('endpoint');
        setSelectedEndpointName(endpoint.name);
        setIsCreateModalOpen(false);
      },
      [refetchEndpoints],
    );

    const reset = useCallback(() => {
      setMode(hasEndpoints ? 'endpoint' : 'direct');
      setSelectedEndpointName(undefined);
      setSelectedPairKey(undefined);
      setManualValues(null);
      setIsManualModalOpen(false);
    }, [hasEndpoints]);

    const isEndpointModeValid = mode === 'endpoint' && Boolean(selectedEndpointName);
    const isDirectModeValid = mode === 'direct' && Boolean(selectedPair);
    const isValid = isEndpointModeValid || Boolean(manualValues) || isDirectModeValid;

    // When the user has entered a model manually, those values take precedence over the pair selection.
    const currentValues = useMemo<ModelSelectionValues>(
      () => manualValues ?? { mode, endpointName: selectedEndpointName, provider, model, apiKeyConfig, saveKey },
      [manualValues, mode, selectedEndpointName, provider, model, apiKeyConfig, saveKey],
    );

    useEffect(() => {
      onValidityChange(isValid);
    }, [isValid, onValidityChange]);

    useEffect(() => {
      onSelectionChange?.(currentValues);
    }, [currentValues, onSelectionChange]);

    useImperativeHandle(
      ref,
      () => ({
        getValues: () => currentValues,
        isValid,
        reset,
      }),
      [currentValues, isValid, reset],
    );

    return (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.lg, paddingTop: theme.spacing.sm }}>
        <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
          <div>
            <Typography.Title level={4} css={{ margin: 0, marginBottom: theme.spacing.xs }}>
              <FormattedMessage
                defaultMessage="Select Model"
                description="Header for the model selection step in issue detection modal"
              />{' '}
              {!showCreateEndpoint && (
                <Tooltip
                  componentId={`${componentId}.endpoint-tip-tooltip`}
                  content={intl.formatMessage({
                    defaultMessage: 'Create an AI Gateway endpoint in AI Gateway → Endpoints tab to reuse it here',
                    description: 'Tooltip suggesting to create an endpoint for reuse',
                  })}
                >
                  <InfoSmallIcon css={{ color: theme.colors.textSecondary, cursor: 'help', verticalAlign: 'middle' }} />
                </Tooltip>
              )}
            </Typography.Title>
            <Typography.Text color="secondary">{description}</Typography.Text>
          </div>

          {/* Model source selector - only show dropdown when there are endpoints */}
          {isLoadingEndpoints ? (
            <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
              <Spinner size="small" />
              <Typography.Text color="secondary">
                <FormattedMessage defaultMessage="Loading endpoints..." description="Loading endpoints message" />
              </Typography.Text>
            </div>
          ) : (
            (hasEndpoints || showCreateEndpoint) && (
              <DialogCombobox
                componentId={`${componentId}.model-source`}
                id={`${componentId}.model-source`}
                value={
                  hasInitializedMode && mode === 'direct' && showConfigureDirectly
                    ? [CONFIGURE_DIRECTLY_VALUE]
                    : selectedEndpointName
                      ? [selectedEndpointName]
                      : []
                }
              >
                <DialogComboboxTrigger
                  withInlineLabel={false}
                  allowClear={false}
                  disabled={readOnly}
                  placeholder={intl.formatMessage({
                    defaultMessage: 'Select endpoint',
                    description: 'Placeholder for endpoint selector',
                  })}
                  renderDisplayedValue={() => (dropdownDisplayValue ? <span>{dropdownDisplayValue}</span> : null)}
                />
                <DialogComboboxContent>
                  <DialogComboboxOptionList>
                    {endpointOptions.length > 0 && (
                      <div
                        css={{
                          maxHeight: 150,
                          overflowY: 'auto',
                          width: '100%',
                          alignSelf: 'stretch',
                        }}
                      >
                        {endpointOptions.map((option) => (
                          <DialogComboboxOptionListSelectItem
                            key={option.value}
                            value={option.value}
                            onChange={() => handleDropdownSelect(option.value)}
                            checked={mode === 'endpoint' && selectedEndpointName === option.value}
                          >
                            {option.label}
                            {option.provider && option.modelName && (
                              <DialogComboboxHintRow>
                                {option.provider} / {option.modelName}
                              </DialogComboboxHintRow>
                            )}
                          </DialogComboboxOptionListSelectItem>
                        ))}
                      </div>
                    )}
                    {(showConfigureDirectly || showCreateEndpoint) && endpointOptions.length > 0 && (
                      <DialogComboboxSeparator />
                    )}
                    {showConfigureDirectly && (
                      <DialogComboboxOptionListSelectItem
                        value={CONFIGURE_DIRECTLY_VALUE}
                        onChange={() => handleDropdownSelect(CONFIGURE_DIRECTLY_VALUE)}
                        checked={hasInitializedMode && mode === 'direct'}
                      >
                        <FormattedMessage
                          defaultMessage="Configure model directly"
                          description="Option to configure model directly instead of using an endpoint"
                        />
                      </DialogComboboxOptionListSelectItem>
                    )}
                    {showCreateEndpoint && (
                      <DialogComboboxOptionListSelectItem
                        value={CREATE_ENDPOINT_VALUE}
                        onChange={() => setIsCreateModalOpen(true)}
                        checked={false}
                      >
                        <FormattedMessage
                          defaultMessage="Create Gateway endpoint"
                          description="Option to create a new AI Gateway endpoint"
                        />
                      </DialogComboboxOptionListSelectItem>
                    )}
                  </DialogComboboxOptionList>
                </DialogComboboxContent>
              </DialogCombobox>
            )
          )}

          {/* Direct mode: single dropdown of pre-allowlisted "provider · model" pairs. */}
          {mode === 'direct' && showConfigureDirectly && (
            <>
              {manualValues && (
                <div
                  css={{
                    display: 'flex',
                    alignItems: 'center',
                    flexWrap: 'wrap',
                    gap: theme.spacing.sm,
                  }}
                >
                  <Typography.Text color="secondary">
                    <FormattedMessage
                      defaultMessage="Using manually configured model: {provider} · {model}"
                      description="Summary of the manually configured model, overriding the allowlisted model dropdown"
                      values={{ provider: manualValues.provider, model: manualValues.model }}
                    />
                  </Typography.Text>
                  <Button
                    componentId={`${componentId}.manual.edit`}
                    type="link"
                    size="small"
                    onClick={() => setIsManualModalOpen(true)}
                  >
                    <FormattedMessage
                      defaultMessage="Edit"
                      description="Link to edit the manually configured model in issue detection"
                    />
                  </Button>
                  <Button
                    componentId={`${componentId}.manual.clear`}
                    type="link"
                    size="small"
                    onClick={() => setManualValues(null)}
                  >
                    <FormattedMessage
                      defaultMessage="Clear"
                      description="Link to clear the manually configured model and return to the allowlisted model dropdown"
                    />
                  </Button>
                </div>
              )}
              {!manualValues &&
                (isLoadingPairs ? (
                  <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
                    <Spinner size="small" />
                    <Typography.Text color="secondary">
                      <FormattedMessage
                        defaultMessage="Loading models..."
                        description="Loading state while fetching allowlisted models for issue detection"
                      />
                    </Typography.Text>
                  </div>
                ) : (
                  <DialogCombobox
                    componentId={`${componentId}.model-pair`}
                    id={`${componentId}.model-pair`}
                    label={intl.formatMessage({
                      defaultMessage: 'Model',
                      description: 'Label for the allowlisted provider/model pair selector',
                    })}
                    value={selectedPair ? [pairKey(selectedPair)] : []}
                  >
                    <DialogComboboxTrigger
                      withInlineLabel={false}
                      allowClear={false}
                      disabled={readOnly}
                      placeholder={intl.formatMessage({
                        defaultMessage: 'Select a model',
                        description: 'Placeholder for the allowlisted provider/model pair selector',
                      })}
                      renderDisplayedValue={() => (selectedPair ? <span>{selectedPair.label}</span> : null)}
                    />
                    <DialogComboboxContent>
                      <DialogComboboxOptionList>
                        {allowlistedPairs.length > 0 ? (
                          <>
                            <DialogComboboxOptionListSearch
                              controlledValue={pairSearch}
                              setControlledValue={setPairSearch}
                            >
                              {filteredPairs.length > 0 ? (
                                filteredPairs.map((pair) => {
                                  const key = pairKey(pair);
                                  return (
                                    <DialogComboboxOptionListSelectItem
                                      key={key}
                                      value={key}
                                      onChange={() => setSelectedPairKey(key)}
                                      checked={selectedPairKey === key}
                                    >
                                      {pair.label}
                                      <DialogComboboxHintRow>{pair.secretName}</DialogComboboxHintRow>
                                    </DialogComboboxOptionListSelectItem>
                                  );
                                })
                              ) : (
                                <DialogComboboxOptionListSelectItem
                                  value=""
                                  onChange={() => {}}
                                  checked={false}
                                  disabled
                                >
                                  <FormattedMessage
                                    defaultMessage="No matching models"
                                    description="Empty state when the model search matches no allowlisted models"
                                  />
                                </DialogComboboxOptionListSelectItem>
                              )}
                            </DialogComboboxOptionListSearch>
                            <DialogComboboxSeparator />
                            <button
                              type="button"
                              css={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: theme.spacing.xs,
                                width: '100%',
                                padding: `${theme.spacing.xs}px ${theme.spacing.md}px`,
                                background: 'none',
                                border: 'none',
                                cursor: 'pointer',
                                textAlign: 'left',
                                color: theme.colors.actionPrimaryBackgroundDefault,
                              }}
                              onClick={() => navigate(CONNECTIONS_ROUTE)}
                            >
                              <PlusIcon />
                              <FormattedMessage
                                defaultMessage="Manage connections"
                                description="Footer link in the model dropdown that navigates to the LLM Connections settings"
                              />
                            </button>
                          </>
                        ) : (
                          <div
                            css={{
                              display: 'flex',
                              flexDirection: 'column',
                              gap: theme.spacing.sm,
                              padding: `${theme.spacing.sm}px ${theme.spacing.md}px`,
                            }}
                          >
                            <Typography.Text color="secondary">
                              <FormattedMessage
                                defaultMessage="No models available. Add a connection with allowed models to get started."
                                description="Empty state shown in the model dropdown when no allowlisted models exist"
                              />
                            </Typography.Text>
                            <button
                              type="button"
                              css={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: theme.spacing.xs,
                                background: 'none',
                                border: 'none',
                                cursor: 'pointer',
                                padding: 0,
                                color: theme.colors.actionPrimaryBackgroundDefault,
                              }}
                              onClick={() => navigate(CONNECTIONS_ROUTE)}
                            >
                              <PlusIcon />
                              <FormattedMessage
                                defaultMessage="Add a connection"
                                description="Action in the empty model dropdown that navigates to the LLM Connections settings"
                              />
                            </button>
                          </div>
                        )}
                      </DialogComboboxOptionList>
                    </DialogComboboxContent>
                  </DialogCombobox>
                ))}
              {!manualValues && (
                <a
                  css={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: theme.spacing.xs,
                    alignSelf: 'flex-start',
                  }}
                  onClick={(e) => {
                    e.preventDefault();
                    navigate(CONNECTIONS_ROUTE);
                  }}
                  href={CONNECTIONS_ROUTE}
                >
                  <FormattedMessage
                    defaultMessage="Manage connections"
                    description="Footer link below the model selector navigating to LLM Connections settings"
                  />
                </a>
              )}
              {!manualValues && (
                <Button
                  componentId={`${componentId}.manual.open`}
                  type="link"
                  size="small"
                  css={{ alignSelf: 'flex-start' }}
                  onClick={() => setIsManualModalOpen(true)}
                >
                  <FormattedMessage
                    defaultMessage="Enter model details manually"
                    description="Link below the model selector that opens a modal to configure a model manually"
                  />
                </Button>
              )}
            </>
          )}
        </div>

        {showCreateEndpoint && (
          <CreateEndpointModal
            open={isCreateModalOpen}
            onClose={() => setIsCreateModalOpen(false)}
            onSuccess={handleCreateEndpointSuccess}
          />
        )}

        <ManualModelConfigModal
          open={isManualModalOpen}
          onClose={() => setIsManualModalOpen(false)}
          onSave={(v) => setManualValues(v)}
          initialValues={manualValues}
          componentId={`${componentId}.manual`}
        />
      </div>
    );
  },
);

GenAIModelSelection.displayName = 'GenAIModelSelection';
