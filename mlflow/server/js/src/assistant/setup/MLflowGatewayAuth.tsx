import { useCallback, useEffect, useState } from 'react';
import {
  Alert,
  Button,
  Empty,
  RefreshIcon,
  Spinner,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { EndpointSelector } from '../../experiment-tracking/components/EndpointSelector';
import { useEndpointsQuery } from '../../gateway/hooks/useEndpointsQuery';
import { GatewayRoutePaths } from '../../gateway/routes';
import { updateConfig } from '../AssistantService';
import { useAssistantConfigQuery } from '../hooks/useAssistantConfigQuery';
import { GATEWAY_PROVIDER_ID } from '../constants';

interface MLflowGatewayAuthProps {
  onBack: () => void;
  onContinue: () => void;
}

export const MLflowGatewayAuth = ({ onBack, onContinue }: MLflowGatewayAuthProps) => {
  const { theme } = useDesignSystemTheme();
  const { config } = useAssistantConfigQuery();
  // TODO: filter to chat-capable endpoints once the gateway schema exposes a
  // task / endpoint_type field. Today the gateway is model-based and does
  // not surface task type, so non-chat endpoints can be selected and will
  // 4xx at chat-time. EndpointSelector renders the model name so users
  // have *some* signal, but it's not a substitute for type-aware filtering.
  const { data: endpoints, isLoading, error, refetch } = useEndpointsQuery();
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>('');

  // Restore previously-selected endpoint from config on first load.
  // `"default"` is the server-side placeholder ProviderConfig.model gets
  // initialized to when the gateway is first selected — treat it as unset
  // so EndpointSelector doesn't render it as a deleted endpoint.
  useEffect(() => {
    const configured = config?.providers?.[GATEWAY_PROVIDER_ID]?.model;
    if (configured && configured !== 'default') {
      setSelectedEndpoint(configured);
    }
  }, [config]);

  // Auto-select the first available endpoint when nothing is selected yet.
  useEffect(() => {
    if (!selectedEndpoint && endpoints.length > 0) {
      setSelectedEndpoint(endpoints[0].name);
    }
  }, [endpoints, selectedEndpoint]);

  const handleContinue = useCallback(async () => {
    if (!selectedEndpoint) {
      return;
    }
    await updateConfig({
      providers: { [GATEWAY_PROVIDER_ID]: { model: selectedEndpoint, selected: true } },
    });
    onContinue();
  }, [onContinue, selectedEndpoint]);

  const openCreateEndpoint = useCallback(() => {
    // MLflow uses hash routing, so SPA routes must be prefixed with `/#`
    // for fresh-tab loads to land on the right page.
    window.open(`/#${GatewayRoutePaths.createEndpointPage}`, '_blank', 'noopener');
  }, []);

  let content: React.ReactNode;
  if (isLoading) {
    content = (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: theme.spacing.lg * 2,
          gap: theme.spacing.md,
        }}
      >
        <Spinner size="default" />
        <Typography.Text color="secondary">Loading gateway endpoints...</Typography.Text>
      </div>
    );
  } else if (error) {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Alert
          componentId="mlflow.assistant.setup.gateway.endpoints_error"
          type="error"
          message={error.message || 'Failed to load endpoints'}
        />
        <div>
          <Button componentId="mlflow.assistant.setup.gateway.refresh" icon={<RefreshIcon />} onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      </div>
    );
  } else if (endpoints.length === 0) {
    content = (
      <div
        css={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: 240,
          width: '100%',
          '& > div': {
            height: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          },
        }}
      >
        <Empty
          title="No gateway endpoints"
          description="Create a chat endpoint on the MLflow AI Gateway, then come back and refresh."
          button={
            <div css={{ display: 'flex', gap: theme.spacing.sm, justifyContent: 'center' }}>
              <Button
                componentId="mlflow.assistant.setup.gateway.create_endpoint"
                type="primary"
                onClick={openCreateEndpoint}
              >
                Create endpoint
              </Button>
              <Button
                componentId="mlflow.assistant.setup.gateway.refresh"
                icon={<RefreshIcon />}
                onClick={() => refetch()}
              >
                Refresh
              </Button>
            </div>
          }
        />
      </div>
    );
  } else {
    content = (
      <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
        <Typography.Text>Select a gateway endpoint:</Typography.Text>
        <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
          <div css={{ flex: 1 }}>
            <EndpointSelector
              currentEndpointName={selectedEndpoint}
              onEndpointSelect={setSelectedEndpoint}
              showCreateButton={false}
              componentIdPrefix="mlflow.assistant.setup.gateway"
            />
          </div>
          <Button
            componentId="mlflow.assistant.setup.gateway.refresh"
            icon={<RefreshIcon />}
            onClick={() => refetch()}
            aria-label="Refresh endpoints"
          />
        </div>
        <Typography.Text color="secondary" css={{ fontSize: theme.typography.fontSizeSm }}>
          Don't see your endpoint?{' '}
          <Typography.Link componentId="mlflow.assistant.setup.gateway.create_link" onClick={openCreateEndpoint}>
            Create a new one
          </Typography.Link>{' '}
          and then refresh.
        </Typography.Text>
      </div>
    );
  }

  const continueDisabled = isLoading || !selectedEndpoint || !endpoints.some((e) => e.name === selectedEndpoint);

  return (
    <div css={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div css={{ flex: 1 }}>{content}</div>
      <div
        css={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: theme.spacing.lg,
          paddingTop: theme.spacing.md,
          borderTop: `1px solid ${theme.colors.border}`,
        }}
      >
        <Button componentId="mlflow.assistant.setup.connection.back" onClick={onBack}>
          Back
        </Button>
        <Button
          componentId="mlflow.assistant.setup.connection.continue"
          type="primary"
          onClick={handleContinue}
          disabled={continueDisabled}
        >
          Continue
        </Button>
      </div>
    </div>
  );
};
