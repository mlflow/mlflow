import { jest, describe, beforeEach, afterEach, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';
import { QueryClient, QueryClientProvider } from '../../query-client/queryClient';

import { ModelTraceExplorerOSSNotebookRenderer } from './ModelTraceExplorerOSSNotebookRenderer';
import { getTraceArtifact } from './mlflow-fetch-utils';
import { MOCK_TRACE } from '../ModelTraceExplorer.test-utils';

// This test renders the full ModelTraceExplorer, which is a heavy component. The configured
// React Testing Library asyncUtilTimeout (10s) exceeds Jest's default 5s per-test timeout, so on
// slow CI workers the findBy* polls get killed by Jest before they can resolve. Raise the timeout
// to match the sibling ModelTraceExplorer.test.tsx convention.
// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(30000);

jest.mock('./mlflow-fetch-utils', () => ({
  getTraceArtifact: jest.fn(),
}));

jest.mock('../hooks/useGetModelTraceInfo', () => ({
  useGetModelTraceInfo: jest.fn().mockReturnValue({
    refetch: jest.fn(),
  }),
}));

describe('ModelTraceExplorerOSSNotebookRenderer', () => {
  beforeEach(() => {
    // eslint-disable-next-line @databricks/no-mock-location -- TODO(FEINF-4390)
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
