import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { createMLflowRoutePath } from '../../../common/utils/RoutingUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import {
  ExperimentPageRevampProvider,
  useExperimentPageRevampContext,
} from '../experiment-page-tabs/ExperimentPageRevampContext';
import { useGetDatasetQuery } from './hooks/useDatasetsQueries';
import { ExperimentEvaluationDatasetDetailPage } from './ExperimentEvaluationDatasetDetailPage';

jest.mock('./hooks/useDatasetsQueries', () => ({
  useGetDatasetQuery: jest.fn(),
}));

jest.mock('./components/DatasetDetailPageContent', () => ({
  DatasetDetailPageContent: () => <div>Dataset detail content</div>,
}));

// Reads the raw context state, so a publish that ignores the shell gate shows up even though readers also check it.
const PublishedTitle = () => (
  <div data-testid="published-title">{useExperimentPageRevampContext().publishedTitle?.title ?? ''}</div>
);

describe('ExperimentEvaluationDatasetDetailPage', () => {
  const { history } = setupTestRouter();

  beforeEach(() => {
    jest.mocked(useGetDatasetQuery).mockReturnValue({
      isLoading: false,
      error: null,
      data: { dataset_id: 'd-1', create_time: '2026-01-01T00:00:00Z', name: 'Friendly dataset' },
    } as any);
  });

  const renderPage = (shellEnabled: boolean) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <TestRouter
            history={history}
            initialEntries={[createMLflowRoutePath('/experiments/1/datasets/d-1')]}
            routes={[
              testRoute(
                <ExperimentPageRevampProvider enabled={shellEnabled}>
                  <PublishedTitle />
                  <ExperimentEvaluationDatasetDetailPage />
                </ExperimentPageRevampProvider>,
                createMLflowRoutePath('/experiments/:experimentId/datasets/:datasetId'),
              ),
            ]}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

  test('publishes the dataset name for the breadcrumb when the page shell is on', async () => {
    renderPage(true);

    expect(await screen.findByText('Dataset detail content')).toBeInTheDocument();
    expect(screen.getByTestId('published-title')).toHaveTextContent('Friendly dataset');
  });

  test('publishes nothing when the page shell is off', async () => {
    renderPage(false);

    expect(await screen.findByText('Dataset detail content')).toBeInTheDocument();
    expect(screen.getByTestId('published-title')).toHaveTextContent('');
  });
});
