import { jest, describe, beforeEach, afterEach, expect, it } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider, useIntl } from '@databricks/i18n';
import { getUser, useUserIdentity } from '../../global-settings/getUser';
import type { ModelTrace } from '../../model-trace-explorer/ModelTrace.types';
import type { UseQueryResult } from '../../query-client/queryClient';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { GenAiEvaluationTracesReview } from './GenAiEvaluationTracesReview';
import { createTestTrace } from '../test-fixtures/EvaluatedTraceTestUtils';
import type { RunEvaluationTracesDataEntry } from '../types';
import { getAssessmentInfos } from '../utils/AggregationUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(120000); // This is quite expensive test

// Mock necessary modules
jest.mock('../../global-settings/getUser', () => ({
  ...jest.requireActual<typeof import('../../global-settings/getUser')>('../../global-settings/getUser'),
  getUser: jest.fn(),
  useUserIdentity: jest.fn(() => ({ user: 'test.user@mlflow.org', isResolved: true })),
}));

const testRunUuid = 'test-run-uuid';

describe('Evaluations review single eval - integration test', () => {
  beforeEach(() => {
    // Mock user ID
    jest.mocked(getUser).mockImplementation(() => 'test.user@mlflow.org');
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'test.user@mlflow.org', isResolved: true });

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

    const renderResult = render(
      <IntlProvider locale="en">
        <TestComponent />
      </IntlProvider>,
    );
    return {
      ...renderResult,
      rerender: () =>
        renderResult.rerender(
          <IntlProvider locale="en">
            <TestComponent />
          </IntlProvider>,
        ),
    };
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

  it('discards pending assessments when the signed-in user changes', async () => {
    const testTrace = createTestTrace({ requestId: 'request_0', request: 'Hello world', assessments: [] });

    const { rerender } = renderTestComponent(testTrace);
    await waitFor(() => expect(screen.getByText('Hello world')).toBeInTheDocument());

    await userEvent.click(screen.getAllByRole('button', { name: 'Add assessment' })[0]);
    await userEvent.click(screen.getByPlaceholderText('Select or type an assessment'));
    await userEvent.paste('helpful');
    // Moving focus out of the combobox commits the custom value.
    await userEvent.click(screen.getByPlaceholderText('Add rationale (optional)'));
    await userEvent.click(screen.getByRole('button', { name: 'Confirm' }));
    expect(screen.getByText('1 pending change')).toBeInTheDocument();

    // The draft is stamped with the previous user's source ID, so it must not be saved by the new user.
    jest.mocked(useUserIdentity).mockReturnValue({ user: 'other.user@mlflow.org', isResolved: true });
    rerender();

    expect(screen.queryByText('1 pending change')).not.toBeInTheDocument();
  });

  it('closes an open assessment form when the signed-in user changes', async () => {
    const testTrace = createTestTrace({ requestId: 'request_0', request: 'Hello world', assessments: [] });

    const { rerender } = renderTestComponent(testTrace);
    await waitFor(() => expect(screen.getByText('Hello world')).toBeInTheDocument());

    await userEvent.click(screen.getAllByRole('button', { name: 'Add assessment' })[0]);
    await userEvent.click(screen.getByPlaceholderText('Select or type an assessment'));
    await userEvent.paste('typed by the previous user');
    expect(screen.getByRole('button', { name: 'Confirm' })).toBeInTheDocument();

    jest.mocked(useUserIdentity).mockReturnValue({ user: 'other.user@mlflow.org', isResolved: true });
    rerender();

    await waitFor(() => expect(screen.queryByRole('button', { name: 'Confirm' })).not.toBeInTheDocument());
    expect(screen.queryByPlaceholderText('Select or type an assessment')).not.toBeInTheDocument();
  });
});
