import type { ComponentProps } from 'react';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { ModelsCellRenderer } from './ModelsCellRenderer';
import { BrowserRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import type { LoggedModelProto } from '../../../../../types';
import { QueryClientProvider, QueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { shouldUseGetLoggedModelsBatchAPI } from '../../../../../../common/utils/FeatureUtils';

jest.mock('../../../../../../common/utils/FeatureUtils', () => ({
  shouldUnifyLoggedModelsAndRegisteredModels: jest.fn(),
  shouldUseGetLoggedModelsBatchAPI: jest.fn(),
}));

// Utility function to get a link by its text content
const getLinkByTextContent = (expectedText: string) =>
  screen.getByText((_, element) => element instanceof HTMLAnchorElement && element?.textContent === expectedText);

describe('ModelsCellRenderer', () => {
  beforeEach(() => {
    jest.mocked(shouldUseGetLoggedModelsBatchAPI).mockReturnValue(true);
  });

  const renderTestComponent = (props: ComponentProps<typeof ModelsCellRenderer>) => {
    const queryClient = new QueryClient();
    // Create a mock provider that supplies the registered versions without mocking the hook
    return render(
      <BrowserRouter>
        <QueryClientProvider client={queryClient}>
          <IntlProvider locale="en">
            <ModelsCellRenderer {...props} />
          </IntlProvider>
        </QueryClientProvider>
      </BrowserRouter>,
    );
  };

  test('renders empty placeholder when no value is provided', () => {
    renderTestComponent({ value: undefined } as any);
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders empty placeholder when no models are available', () => {
    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  test('renders legacy registered models correctly', () => {
    renderTestComponent({
      value: {
        registeredModels: [{ name: 'Model1', version: '1', source: '' } as any],
        loggedModels: [],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    expect(screen.getByText('Model1')).toBeInTheDocument();
    expect(screen.getByText('v1')).toBeInTheDocument();
  });

  test('renders V3 logged models correctly when they have no registered versions', () => {
    const loggedModelV3: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3',
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    expect(screen.getByText('LoggedModelV3')).toBeInTheDocument();
  });

  test('hides V3 logged models when they have associated registered versions', () => {
    const loggedModelV3: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([
              {
                name: 'RegisteredFromV3',
                version: '2',
              },
            ]),
          },
        ],
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // The logged model should be hidden, but its registered version should be visible
    expect(screen.queryByText('LoggedModelV3')).not.toBeInTheDocument();
    expect(screen.getByText('RegisteredFromV3')).toBeInTheDocument();
    expect(screen.getByText('v2')).toBeInTheDocument();
  });

  test('renders multiple legacy models correctly', async () => {
    renderTestComponent({
      value: {
        registeredModels: [
          { name: 'Model2', version: '2', source: '/artifacts/model2/2' } as any,
          { name: 'Model1', version: '1', source: '/artifacts/model1/1' } as any,
        ],
        loggedModels: [],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // We should see overflow with +1 element available
    await userEvent.click(screen.getByText('+1'));

    expect(getLinkByTextContent('Model2 v2')).toBeInTheDocument();
    expect(getLinkByTextContent('Model1 v1')).toBeInTheDocument();
  });

  test('renders unregistered logged models with flavor name', () => {
    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [{ artifactPath: 'artifacts/model1', flavors: ['sklearn'], utcTimeCreated: 12345 }],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    expect(screen.getByText('sklearn')).toBeInTheDocument();
  });

  test('renders default "Model" text when no flavor is available', () => {
    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [{ artifactPath: 'artifacts/model1', flavors: [], utcTimeCreated: 12345 }],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    expect(screen.getByText('Model')).toBeInTheDocument();
  });

  test('merges legacy logged models and registered models correctly', async () => {
    renderTestComponent({
      value: {
        registeredModels: [{ name: 'RegisteredModel', version: '3', source: '/artifacts/123' } as any],
        loggedModels: [
          { artifactPath: 'artifacts/model1', flavors: ['sklearn'], utcTimeCreated: 12345 },
          { artifactPath: 'models/model2', flavors: ['pytorch'], utcTimeCreated: 12346 },
        ],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // We should see overflow with +2 elements available
    await userEvent.click(screen.getByText('+2'));

    // Should show both the registered model and the logged models
    expect(getLinkByTextContent('RegisteredModel v3')).toBeInTheDocument();
    expect(getLinkByTextContent('sklearn')).toBeInTheDocument();
    expect(getLinkByTextContent('pytorch')).toBeInTheDocument();
  });

  test('handles legacy logged models with matching registered models', () => {
    // This test simulates a logged model that has been registered
    renderTestComponent({
      value: {
        registeredModels: [
          {
            name: 'RegisteredModel',
            version: '3',
            source: 'xyz/artifacts/model1',
          } as any,
        ],
        loggedModels: [
          {
            artifactPath: 'model1',
            flavors: ['sklearn'],
            utcTimeCreated: 12345,
          },
        ],
        loggedModelsV3: [],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // Should show the registered model once (not duplicated)
    expect(getLinkByTextContent('RegisteredModel v3')).toBeInTheDocument();

    // The flavor name should not be visible since the model is registered
    expect(screen.queryByText('sklearn')).not.toBeInTheDocument();

    // We should not see the "+" button since there are no additional models
    expect(screen.queryByRole('button', { name: /\+\d/ })).not.toBeInTheDocument();
  });

  test('renders both legacy registered models and V3 logged models together', async () => {
    const loggedModelV3: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3',
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [{ name: 'RegisteredModel', version: '2', source: '/artifacts/legacy_model' } as any],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // We should see overflow with +1 elements available
    await userEvent.click(screen.getByText('+1'));
    expect(getLinkByTextContent('RegisteredModel v2')).toBeInTheDocument();
    expect(getLinkByTextContent('LoggedModelV3')).toBeInTheDocument();
  });

  test('renders correctly when both legacy registered models and registered versions from V3 models exist', async () => {
    const loggedModelV3: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([
              {
                name: 'RegisteredFromV3',
                version: '2',
              },
            ]),
          },
        ],
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [{ name: 'DirectlyRegistered', version: '5', source: '/artifacts/legacy_model' } as any],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // We should see overflow with +1 elements available
    await userEvent.click(screen.getByText('+1'));
    expect(getLinkByTextContent('DirectlyRegistered v5')).toBeInTheDocument();
    expect(getLinkByTextContent('RegisteredFromV3 v2')).toBeInTheDocument();

    // The original logged model should be hidden
    expect(screen.queryByText('LoggedModelV3')).not.toBeInTheDocument();
  });

  test('handles multiple V3 logged models with different registered versions', async () => {
    const loggedModelV3A: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3A',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3-a',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([
              {
                name: 'RegisteredFromV3A',
                version: '2',
              },
            ]),
          },
        ],
      },
    };

    const loggedModelV3B: LoggedModelProto = {
      info: {
        model_id: 'model-id-2',
        name: 'LoggedModelV3B',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3-b',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([
              {
                name: 'RegisteredFromV3B',
                version: '4',
              },
            ]),
          },
        ],
      },
    };

    const loggedModelV3C: LoggedModelProto = {
      info: {
        model_id: 'model-id-3',
        name: 'LoggedModelV3C',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3-c',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([]), // No registered version for this model
          },
        ],
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3A, loggedModelV3B, loggedModelV3C],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    await userEvent.click(screen.getByText('+2'));
    expect(getLinkByTextContent('RegisteredFromV3A v2')).toBeInTheDocument();
    expect(getLinkByTextContent('RegisteredFromV3B v4')).toBeInTheDocument();

    // LoggedModelV3B should be visible since it has no registered version
    expect(screen.getByText('LoggedModelV3C')).toBeInTheDocument();
  });

  test('handles V3 logged models with multiple registered versions', async () => {
    const loggedModelV3: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([
              {
                name: 'RegisteredModel',
                version: '1',
              },
              {
                name: 'RegisteredModel',
                version: '2',
              },
            ]),
          },
        ],
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    // In total, we should have 2 models so "+1" should be visible
    await userEvent.click(screen.getByText('+1'));

    expect(getLinkByTextContent('RegisteredModel v1')).toBeInTheDocument();
    expect(getLinkByTextContent('RegisteredModel v2')).toBeInTheDocument();

    // The logged model should be hidden
    expect(screen.queryByText('LoggedModelV3')).not.toBeInTheDocument();
  });

  test('should not unfurl logged models into registered models when feature flag is off', async () => {
    jest.mocked(shouldUseGetLoggedModelsBatchAPI).mockReturnValue(false);

    const loggedModelV3: LoggedModelProto = {
      info: {
        model_id: 'model-id-1',
        name: 'LoggedModelV3',
        experiment_id: 'exp-1',
        artifact_uri: 'artifacts/model-v3',
        tags: [
          {
            key: 'mlflow.modelVersions',
            value: JSON.stringify([
              {
                name: 'RegisteredModel',
                version: '1',
              },
              {
                name: 'RegisteredModel',
                version: '2',
              },
            ]),
          },
        ],
      },
    };

    renderTestComponent({
      value: {
        registeredModels: [],
        loggedModels: [],
        loggedModelsV3: [loggedModelV3],
        experimentId: 'exp-1',
        runUuid: 'run-1',
      },
    });

    expect(screen.getByText('LoggedModelV3')).toBeInTheDocument();
    // We should not see the "+" button since there are no additional models
    expect(screen.queryByRole('button', { name: /\+\d/ })).not.toBeInTheDocument();
  });
});
