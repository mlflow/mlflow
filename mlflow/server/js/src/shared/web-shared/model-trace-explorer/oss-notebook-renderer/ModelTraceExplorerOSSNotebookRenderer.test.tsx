import { jest, describe, beforeEach, afterEach, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { ModelTraceExplorerOSSNotebookRenderer } from './ModelTraceExplorerOSSNotebookRenderer';
import { getTraceArtifact } from './mlflow-fetch-utils';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';

jest.mock('./mlflow-fetch-utils', () => ({
  getTraceArtifact: jest.fn(),
}));

jest.mock('../hooks/useGetModelTraceInfo', () => ({
  useGetModelTraceInfo: jest.fn().mockReturnValue({
    refetch: jest.fn(),
  }),
}));

const renderComponent = () => {
  const queryClient = new QueryClient();
  return render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <QueryClientProvider client={queryClient}>
          <ModelTraceExplorerOSSNotebookRenderer />
        </QueryClientProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );
};

describe('ModelTraceExplorerOSSNotebookRenderer', () => {
  beforeEach(() => {
    Object.defineProperty(window, 'location', {
      configurable: true,
      writable: true,
      value: new URL('http://localhost:5000/?trace_id=1&experiment_id=1'),
    });

    jest.mocked(getTraceArtifact).mockResolvedValue(MOCK_TRACE);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders without crashing', async () => {
    renderComponent();

    expect(await screen.findByText('MLflow Trace UI')).toBeInTheDocument();
    expect(await screen.findByText('document-qa-chain')).toBeInTheDocument();
  });

  it('includes workspace param in "View in MLflow UI" link when workspace is present in URL', async () => {
    Object.defineProperty(window, 'location', {
      configurable: true,
      writable: true,
      value: new URL('http://localhost:5000/?trace_id=tr-123&experiment_id=42&workspace=my-workspace'),
    });

    renderComponent();

    const link = await screen.findByTitle('View in MLflow UI');
    expect(link).toHaveAttribute(
      'href',
      '/#/experiments/42/traces?selectedEvaluationId=tr-123&compareRunsMode=TRACES&workspace=my-workspace',
    );
  });

  it('omits workspace param in "View in MLflow UI" link when workspace is absent from URL', async () => {
    renderComponent();

    const link = await screen.findByTitle('View in MLflow UI');
    expect(link).toHaveAttribute('href', '/#/experiments/1/traces?selectedEvaluationId=1&compareRunsMode=TRACES');
  });
});
