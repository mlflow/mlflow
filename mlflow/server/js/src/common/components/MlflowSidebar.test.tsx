import { describe, jest, test, expect } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MlflowSidebar } from './MlflowSidebar';
import { MemoryRouter, Outlet, Route, Routes, useNavigate } from '../utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../utils/reactQueryHooks';
import { renderWithDesignSystem } from '../utils/TestUtils.react18';

jest.mock('../../account/hooks', () => ({
  useCurrentUserQuery: jest.fn(() => ({ data: undefined, isError: true, isLoading: false })),
  useCurrentUserIsAdmin: jest.fn(() => false),
  useIsBasicAuth: jest.fn(() => false),
}));

jest.mock('../../experiment-tracking/hooks/useServerInfo', () => ({
  ...jest.requireActual<typeof import('../../experiment-tracking/hooks/useServerInfo')>(
    '../../experiment-tracking/hooks/useServerInfo',
  ),
  useFeatureEnabled: jest.fn(() => false),
}));

jest.mock('./MlflowSidebarExperimentItems', () => ({
  MlflowSidebarExperimentItems: ({ experimentId }: { experimentId?: string }) => (
    <div data-testid="nested-experiment-items">{experimentId}</div>
  ),
}));

const GoBackButton = () => {
  const navigate = useNavigate();
  return <button onClick={() => navigate(-1)}>Go back</button>;
};

describe('MlflowSidebar', () => {
  const renderTestComponent = (initialEntries: string[]) => {
    const queryClient = new QueryClient();
    return renderWithDesignSystem(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={initialEntries} initialIndex={initialEntries.length - 1}>
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <MlflowSidebar showSidebar setShowSidebar={() => {}} />
                  <Outlet />
                </>
              }
            >
              <Route path="experiments" element={<div>Experiments list</div>} />
              <Route path="experiments/:experimentId/*" element={<GoBackButton />} />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );
  };

  test('shows top-level items after navigating back from an experiment to the experiments list', async () => {
    renderTestComponent(['/experiments', '/experiments/123/traces']);

    expect(screen.getByTestId('nested-experiment-items')).toHaveTextContent('123');

    await userEvent.click(screen.getByText('Go back'));

    expect(screen.getByText('Experiments list')).toBeInTheDocument();
    expect(screen.queryByTestId('nested-experiment-items')).not.toBeInTheDocument();
    expect(screen.getByText('Home')).toBeInTheDocument();
  });
});
