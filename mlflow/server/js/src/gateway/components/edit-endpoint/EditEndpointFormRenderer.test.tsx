import React from 'react';
import { describe, expect, jest, test } from '@jest/globals';
import { useForm } from 'react-hook-form';
import { MemoryRouter } from '../../../common/utils/RoutingUtils';
import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import { EditEndpointFormRenderer } from './EditEndpointFormRenderer';
import type { EditEndpointFormData } from '../../hooks/useEditEndpointForm';
import type { Endpoint } from '../../types';

jest.mock('./TrafficSplitConfigurator', () => ({ TrafficSplitConfigurator: () => null }));
jest.mock('./FallbackModelsConfigurator', () => ({ FallbackModelsConfigurator: () => null }));
jest.mock('./StarterCodeCard', () => ({ StarterCodeCard: () => null }));
jest.mock('./EditableEndpointName', () => ({ EditableEndpointName: () => null }));
jest.mock('./GatewayUsageSection', () => ({ GatewayUsageSection: () => null }));
jest.mock('../guardrails/GuardrailsTabContent', () => ({ GuardrailsTabContent: () => null }));
jest.mock('../../../common/components/long-form/LongFormSummary', () => ({
  LongFormSummary: ({ children }: any) => children,
}));
jest.mock('../../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3Logs', () => ({
  TracesV3Logs: () => null,
}));
jest.mock('../../../experiment-tracking/hooks/useMonitoringConfig', () => ({
  MonitoringConfigProvider: ({ children }: any) => children,
}));
jest.mock('../../../experiment-tracking/hooks/useMonitoringFilters', () => ({
  useMonitoringFiltersTimeRange: () => ({ key: 'LAST_1_HOUR' }),
}));
jest.mock('../../../experiment-tracking/components/experiment-page/components/traces-v3/TracesV3DateSelector', () => ({
  TracesV3DateSelector: () => null,
}));

const endpoint: Endpoint = {
  endpoint_id: 'ep-1',
  name: 'test-endpoint',
  created_at: 1735689600000,
  last_updated_at: 1735689600000,
  model_mappings: [],
};

const TestHarness = ({
  experimentId,
  initialEntry,
  trafficSplitModels = [],
  fallbackModels = [],
  hasChanges = false,
}: {
  experimentId: string;
  initialEntry: string;
  trafficSplitModels?: EditEndpointFormData['trafficSplitModels'];
  fallbackModels?: EditEndpointFormData['fallbackModels'];
  hasChanges?: boolean;
}) => {
  const form = useForm<EditEndpointFormData>({
    defaultValues: {
      name: endpoint.name,
      trafficSplitModels,
      fallbackModels,
      usageTracking: Boolean(experimentId),
      experimentId,
    },
  });

  return (
    <MemoryRouter initialEntries={[initialEntry]}>
      <EditEndpointFormRenderer
        form={form}
        isLoadingEndpoint={false}
        isSubmitting={false}
        loadError={null}
        mutationError={null}
        errorMessage={null}
        endpoint={endpoint}
        existingEndpoints={[endpoint]}
        isFormComplete
        hasChanges={hasChanges}
        onSubmit={jest.fn(async () => {})}
        onCancel={jest.fn()}
        onNameUpdate={jest.fn(async () => {})}
        onUsageTrackingUpdate={jest.fn(async () => {})}
      />
    </MemoryRouter>
  );
};

describe('EditEndpointFormRenderer', () => {
  test('prevents saving a TypeSafe endpoint with a chat model fallback', () => {
    const model = {
      modelDefinitionName: 'jev',
      provider: 'typesafe',
      modelName: 'jev-latest',
      secretMode: 'existing' as const,
      existingSecretId: 'secret-1',
      newSecret: { name: '', authMode: '', secretFields: {}, configFields: {} },
    };
    renderWithDesignSystem(
      <TestHarness
        experimentId=""
        initialEntry="/?tab=overview"
        trafficSplitModels={[{ ...model, weight: 100 }]}
        fallbackModels={[{ ...model, provider: 'openai', modelName: 'gpt-4o', fallbackOrder: 1 }]}
        hasChanges
      />,
    );

    expect(
      screen.getByText('TypeSafe endpoints require all primary and fallback models to use TypeSafe.'),
    ).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save changes' })).toBeDisabled();
  });

  test('disables Guardrails tab when experiment id is missing', () => {
    renderWithDesignSystem(<TestHarness experimentId="" initialEntry="/?tab=overview" />);
    expect(screen.getByRole('tab', { name: 'Guardrails' })).toBeDisabled();
  });

  test('keeps Guardrails tab enabled when requested directly in URL', () => {
    renderWithDesignSystem(<TestHarness experimentId="" initialEntry="/?tab=guardrails" />);
    expect(screen.getByRole('tab', { name: 'Guardrails' })).not.toBeDisabled();
  });
});
