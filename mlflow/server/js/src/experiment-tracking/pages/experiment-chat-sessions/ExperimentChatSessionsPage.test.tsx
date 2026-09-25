import { describe, it, expect, jest, beforeEach } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { renderWithIntl, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { MemoryRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
import { shouldEnableSessionGrouping } from '@databricks/web-shared/genai-traces-table';

import ExperimentChatSessionsPage from './ExperimentChatSessionsPage';

jest.mock('@databricks/web-shared/genai-traces-table', () => ({
  ...jest.requireActual<typeof import('@databricks/web-shared/genai-traces-table')>(
    '@databricks/web-shared/genai-traces-table',
  ),
  shouldEnableSessionGrouping: jest.fn(() => true),
}));

jest.mock('@mlflow/mlflow/src/common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/RoutingUtils')>(
    '@mlflow/mlflow/src/common/utils/RoutingUtils',
  ),
  useParams: () => ({ experimentId: '123' }),
}));

const renderPage = () =>
  renderWithIntl(
    <DesignSystemProvider>
      <MemoryRouter>
        <ExperimentChatSessionsPage />
      </MemoryRouter>
    </DesignSystemProvider>,
  );

describe('ExperimentChatSessionsPage', () => {
  beforeEach(() => {
    jest.mocked(shouldEnableSessionGrouping).mockReturnValue(true);
  });

  it('shows the moved notice and a link to grouped traces when session grouping is enabled', () => {
    renderPage();

    expect(screen.getByText('The Sessions view has moved to the Traces tab.')).toBeInTheDocument();

    const link = screen.getByRole('link', { name: 'View Sessions in Traces tab →' });
    expect(link).toHaveAttribute('href', expect.stringContaining('/experiments/123/traces'));
    expect(link).toHaveAttribute('href', expect.stringContaining('groupBy=session'));
  });
});
