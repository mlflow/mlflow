import { render, screen, waitFor } from '@testing-library/react';
import type { ComponentProps } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider, useIntl } from '@databricks/i18n';
import { getUser } from '@databricks/web-shared/global-settings';
import type { ModelTrace } from '@databricks/web-shared/model-trace-explorer';
import type { UseQueryResult } from '@databricks/web-shared/query-client';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { GenAiEvaluationTracesReview } from './GenAiEvaluationTracesReview';
import { createTestTrace } from '../test-fixtures/EvaluatedTraceTestUtils';
import type { RunEvaluationTracesDataEntry } from '../types';
import { getAssessmentInfos } from '../utils/AggregationUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(120000); // This is quite expensive test

// Mock necessary modules
jest.mock('@databricks/web-shared/global-settings', () => ({
  getUser: jest.fn(),
}));

const testRunUuid = 'test-run-uuid';

describe('Evaluations review single eval - integration test', () => {
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

  const renderTestComponent = (
    evaluation: RunEvaluationTracesDataEntry,
    additionalProps: Partial<ComponentProps<typeof GenAiEvaluationTracesReview>> = {},
  ) => {
    const TestComponent = () => {
      const intl = useIntl();

      return (
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
            {evaluation ? (
              <GenAiEvaluationTracesReview
                experimentId="test-experiment-id"
                runUuid={testRunUuid}
                evaluation={evaluation}
                selectNextEval={() => {}}
                isNextAvailable={false}
                runDisplayName="Test Run"
                assessmentInfos={getAssessmentInfos(intl, [evaluation], undefined)}
                traceQueryResult={
                  {
                    isLoading: false,
                    data: undefined,
                  } as unknown as UseQueryResult<ModelTrace | undefined, unknown>
                }
                compareToTraceQueryResult={
                  {
                    isLoading: false,
                    data: undefined,
                  } as unknown as UseQueryResult<ModelTrace | undefined, unknown>
                }
                {...additionalProps}
              />
            ) : (
              <></>
            )}
          </QueryClientProvider>
        </DesignSystemProvider>
      );
    };

    return render(
      <IntlProvider locale="en">
        <TestComponent />
      </IntlProvider>,
    );
  };

  const waitForViewToBeReady = () =>
    waitFor(() => expect(screen.getByText(/Overall assessment/)).toBeInTheDocument(), {
      timeout: 2000,
    });

  it('Make sure a single evaluation can be rendered', async () => {
    const testTrace = createTestTrace({
      requestId: 'request_0',
      request: 'Hello world',
      assessments: [
        {
          name: 'overall_assessment',
          value: 'yes',
          dtype: 'pass-fail',
        },
      ],
    });

    renderTestComponent(testTrace);

    await waitForViewToBeReady();

    // Smoke test that the eval rendered.
    expect(screen.getByText('Hello world')).toBeInTheDocument();

    expect(screen.getByText('Detailed assessments')).toBeInTheDocument();
  });
});
