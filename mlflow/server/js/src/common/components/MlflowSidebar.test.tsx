import { describe, jest, test, expect } from '@jest/globals';
import { act, screen } from '@testing-library/react';
import { MlflowSidebar } from './MlflowSidebar';
import { MemoryRouter, Outlet, Route, Routes, useNavigate } from '../utils/RoutingUtils';
import type { NavigateFunction } from '../utils/RoutingUtils';
import { QueryClient, QueryClientProvider } from '../utils/reactQueryHooks';
import { renderWithDesignSystem } from '../utils/TestUtils.react18';

jest.mock('../../account/hooks', () => ({
  useCurrentUserIsAdmin: jest.fn(() => false),
  useCurrentUserQuery: jest.fn(() => ({ data: undefined, isError: true, isLoading: false })),
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
    <li>{`Experiment items for ${experimentId}`}</li>
  ),
}));

describe('MlflowSidebar', () => {
  test('shows the top-level menu after navigating back to the experiments list', () => {
    let navigate: NavigateFunction | undefined;
    const NavigateCapture = () => {
      navigate = useNavigate();
      return null;
    };
    const Layout = () => (
      <>
        <MlflowSidebar showSidebar setShowSidebar={() => {}} />
        <NavigateCapture />
        <Outlet />
      </>
    );

    renderWithDesignSystem(
      <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/experiments', '/experiments/123/traces']} initialIndex={1}>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/experiments" element={null} />
              <Route path="/experiments/:experimentId/*" element={null} />
            </Route>
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    );

    expect(screen.getByText('Experiment items for 123')).toBeInTheDocument();
    expect(screen.queryByText('Home')).not.toBeInTheDocument();

    // Simulate the browser back button
    act(() => navigate?.(-1));

    expect(screen.queryByText('Experiment items for 123')).not.toBeInTheDocument();
    expect(screen.getByText('Home')).toBeInTheDocument();
  });
});
