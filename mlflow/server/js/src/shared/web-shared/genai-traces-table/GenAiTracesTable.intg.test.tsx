import { render, screen, waitFor } from '@testing-library/react';
import type { ComponentProps } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { GenAiTracesTable } from './GenAITracesTable';
// eslint-disable-next-line import/no-namespace
import * as GenAiTracesTableUtils from './GenAiTracesTable.utils';
import type { GenAiEvaluationTracesReview } from './components/GenAiEvaluationTracesReview';
import { createTestTraces } from './test-fixtures/EvaluatedTraceTestUtils';
import type { RunEvaluationTracesDataEntry } from './types';
import { testRoute, TestRouter } from './utils/RoutingTestUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(120000); // This is quite expensive test

// Mock necessary modules
jest.mock('@databricks/web-shared/global-settings', () => ({
  getUser: jest.fn(),
}));

jest.mock('@databricks/web-shared/hooks', () => {
  return {
    ...jest.requireActual<typeof import('@databricks/web-shared/hooks')>('@databricks/web-shared/hooks'),
    getLocalStorageItemByParams: jest.fn().mockReturnValue({ hiddenColumns: undefined }),
    useLocalStorage: jest.fn().mockReturnValue([{}, jest.fn()]),
  };
});

const testRunUuid = 'test-run-uuid';

describe('Evaluations overview - integration test', () => {
  beforeEach(() => {
    // Mock user ID
    jest.mocked(getUser).mockImplementation(() => 'test.user@mlflow.org');

    // Mocked returned timestamp
    jest.spyOn(Date, 'now').mockImplementation(() => 1000000);

    // Silence a noisy issue with Typeahead component and its '_TYPE' prop
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    const originalConsoleError = console.error;
    jest.spyOn(console, 'error').mockImplementation((...args) => {
      if (args[0]?.includes?.('React does not recognize the `%s` prop on a DOM element')) {
        return;
      }
      originalConsoleError(...args);
    });

    // This is necessary to avoid ".scrollTo is not a function" error
    // See https://github.com/vuejs/vue-test-utils/issues/319#issuecomment-354667621
    Element.prototype.scrollTo = () => {};
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  const waitForViewToBeReady = () =>
    waitFor(() => {
      expect(screen.getAllByText(/request/)[0]).toBeInTheDocument();
    });

  it('Make sure a single evaluation can be rendered', async () => {
    const testTraces = createTestTraces([
      {
        requestId: 'request_0',
        request: 'Hello world',
        assessments: [
          {
            name: 'overall_assessment',
            value: 'yes',
            dtype: 'pass-fail',
          },
        ],
      },
    ]);

    renderTestComponent(testTraces);

    await waitForViewToBeReady();

    // Smoke test that the metrics panel rendered.
    expect(screen.getAllByText('Overall')[0]).toBeInTheDocument();
  });

  it('Make sure all dtypes can be rendered', async () => {
    const testTraces = createTestTraces([
      {
        requestId: 'request_0',
        request: 'Hello world',
        assessments: [
          {
            name: 'float_assessment',
            value: 1.5,
            dtype: 'float',
          },
          {
            name: 'boolean_assessment',
            value: true,
            dtype: 'boolean',
          },
          {
            name: 'string_assessment',
            value: 'string_value',
            dtype: 'string',
          },
          {
            name: 'pass_fail_assessment',
            value: 'yes',
            dtype: 'pass-fail',
          },
        ],
      },
      // Make sure we don't error out for null values.
      {
        requestId: 'request_1',
        request: 'Hello world 2',
        assessments: [
          {
            name: 'float_assessment',
            value: undefined,
            dtype: 'float',
          },
          {
            name: 'boolean_assessment',
            value: undefined,
            dtype: 'boolean',
          },
          {
            name: 'string_assessment',
            value: undefined,
            dtype: 'string',
          },
          {
            name: 'pass_fail_assessment',
            value: undefined,
            dtype: 'pass-fail',
          },
        ],
      },
      // Make sure we don't error out for undefined assessments.
      {
        requestId: 'request_2',
        request: 'Hello world 3',
        assessments: [],
      },
    ]);

    renderTestComponent(testTraces);

    await waitForViewToBeReady();

    // Make sure that all assessments are rendered in the header.
    expect(screen.getAllByText('float_assessment')[0]).toBeInTheDocument();
    expect(screen.getAllByText('boolean_assessment')[0]).toBeInTheDocument();
    expect(screen.getAllByText('string_assessment')[0]).toBeInTheDocument();
    expect(screen.getAllByText('pass_fail_assessment')[0]).toBeInTheDocument();

    // Make sure the table has the values above (only 2 rows for each of the traces above).
    expect(screen.getAllByText('1.5').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('True').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('string_value').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Pass').length).toBeGreaterThanOrEqual(1);

    // Make sure the null values are rendered as well.
    expect(screen.getAllByText('null')[0]).toBeInTheDocument();
  });

  it('Make sure comparison renders', async () => {
    const currentTestTraces = createTestTraces([
      {
        requestId: 'request_0',
        request: 'Hello world',
        assessments: [
          {
            name: 'float_assessment',
            value: 1.5,
            dtype: 'float',
          },
          {
            name: 'boolean_assessment',
            value: true,
            dtype: 'boolean',
          },
          {
            name: 'string_assessment',
            value: 'string_value',
            dtype: 'string',
          },
          {
            name: 'pass_fail_assessment',
            value: 'yes',
            dtype: 'pass-fail',
          },
        ],
      },
    ]);
    const compareToTestTraces = createTestTraces([
      // Make sure we don't error out for null values.
      {
        requestId: 'request_1',
        request: 'Hello world 2',
        assessments: [
          {
            name: 'float_assessment',
            value: null,
            dtype: 'float',
          },
          {
            name: 'boolean_assessment',
            value: null,
            dtype: 'boolean',
          },
          {
            name: 'string_assessment',
            value: null,
            dtype: 'string',
          },
          {
            name: 'pass_fail_assessment',
            value: null,
            dtype: 'pass-fail',
          },
        ],
      },
    ]);

    renderTestComponent(currentTestTraces, compareToTestTraces);

    await waitForViewToBeReady();

    // Make sure that all assessments are rendered in the header.
    expect(screen.getAllByText('float_assessment')[0]).toBeInTheDocument();
    expect(screen.getAllByText('boolean_assessment')[0]).toBeInTheDocument();
    expect(screen.getAllByText('string_assessment')[0]).toBeInTheDocument();
    expect(screen.getAllByText('pass_fail_assessment')[0]).toBeInTheDocument();

    // Make sure the table has the values above (only 2 rows for each of the traces above).
    expect(screen.getAllByText('1.5').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('True').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('string_value').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Pass').length).toBeGreaterThanOrEqual(1);

    // Make sure the null values are rendered as well.

    // TODO ML-48427: Investigate why this is failing and re-enable or replace test with updated component
    // expect(screen.getAllByText('null')[0]).toBeInTheDocument();
  });

  const renderTestComponent = (
    currentEvaluationResults: RunEvaluationTracesDataEntry[],
    compareToEvaluationResults: RunEvaluationTracesDataEntry[] = [],
    additionalProps: Partial<ComponentProps<typeof GenAiEvaluationTracesReview>> = {},
  ) => {
    const TestComponent = () => {
      return (
        <TestRouter
          routes={[
            testRoute(
              <DesignSystemProvider>
                <QueryClientProvider
                  client={
                    new QueryClient({
                      logger: {
                        error: () => {},
                        log: () => {},
                        warn: () => {},
                      },
                    })
                  }
                >
                  <GenAiTracesTable
                    experimentId="test-experiment-id"
                    currentRunDisplayName="Test Run"
                    currentEvaluationResults={currentEvaluationResults}
                    compareToEvaluationResults={compareToEvaluationResults}
                    runUuid={testRunUuid}
                    {...(additionalProps || {})}
                  />
                </QueryClientProvider>
              </DesignSystemProvider>,
            ),
          ]}
        />
      );
    };

    return render(
      <IntlProvider locale="en">
        <TestComponent />
      </IntlProvider>,
    );
  };
});
