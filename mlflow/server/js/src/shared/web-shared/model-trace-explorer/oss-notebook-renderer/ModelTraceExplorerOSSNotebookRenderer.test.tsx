import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';

import { ModelTraceExplorerOSSNotebookRenderer } from './ModelTraceExplorerOSSNotebookRenderer';
import { getTraceArtifact } from './mlflow-fetch-utils';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';

jest.mock('./mlflow-fetch-utils', () => ({
  getTraceArtifact: jest.fn(),
}));

jest.mock('../hooks/useGetModelTraceInfoV3', () => ({
  useGetModelTraceInfoV3: jest.fn().mockReturnValue({
    refetch: jest.fn(),
  }),
}));

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
    const queryClient = new QueryClient();

    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <ModelTraceExplorerOSSNotebookRenderer />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );

    expect(await screen.findByText('MLflow Trace UI')).toBeInTheDocument();
    expect(await screen.findByText('document-qa-chain')).toBeInTheDocument();
  });
});
