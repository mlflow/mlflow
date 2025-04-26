import { render, renderHook, screen, waitFor } from '@testing-library/react';
import { useNavigateToModelsPageByDefault } from './useNavigateToModelsPageByDefault';
import { isExperimentLoggedModelsUIEnabled } from '../../../../common/utils/FeatureUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../../common/utils/RoutingTestUtils';

jest.mock('../../../../common/utils/FeatureUtils', () => ({
  isExperimentLoggedModelsUIEnabled: jest.fn(),
}));

// TODO(ML-47062): Re-enable this feature when we have a proper solution.
// eslint-disable-next-line jest/no-disabled-tests
describe.skip('useNavigateToModelsPageByDefault', () => {
  const { history } = setupTestRouter();
  const renderTestHook = (initialRoute: string) => {
    const TestRunsPage = () => {
      const shouldRedirectToModelsPage = useNavigateToModelsPageByDefault();

      if (shouldRedirectToModelsPage) {
        return null;
      }

      return <span>this is runs page</span>;
    };
    const TestLoggedModelsPage = () => <span>this is logged models page</span>;
    return render(
      <TestRouter
        history={history}
        routes={[
          testRoute(<TestRunsPage />, '/ml/experiments/:experimentId'),
          testRoute(<TestRunsPage />, '/ml/compare-experiments/'),
          testRoute(<TestLoggedModelsPage />, '/ml/experiments/:experimentId/models'),
        ]}
        initialEntries={[initialRoute]}
      />,
    );
  };
  test('should not redirect to the models page if the feature is disabled', async () => {
    jest.mocked(isExperimentLoggedModelsUIEnabled).mockImplementation(() => false);

    renderTestHook('/ml/experiments/123');

    await waitFor(() => {
      expect(screen.getByText('this is runs page')).toBeInTheDocument();
    });
  });

  test('should not redirect to the models page if tab is explicitly set', async () => {
    jest.mocked(isExperimentLoggedModelsUIEnabled).mockImplementation(() => true);

    renderTestHook('/ml/experiments/123?compareRunsMode=TABLE');

    await waitFor(() => {
      expect(screen.getByText('this is runs page')).toBeInTheDocument();
    });
  });

  test('should not redirect to the models page if there are multiple experiments compared', async () => {
    jest.mocked(isExperimentLoggedModelsUIEnabled).mockImplementation(() => true);

    renderTestHook('/ml/compare-experiments/?experiments=' + encodeURIComponent(JSON.stringify(['123', '456'])));

    await waitFor(() => {
      expect(screen.getByText('this is runs page')).toBeInTheDocument();
    });
  });

  test('should redirect to the models page if feature is enabled, only one experiment is enabled and there is no tab set in the URL', async () => {
    jest.mocked(isExperimentLoggedModelsUIEnabled).mockImplementation(() => true);

    renderTestHook('/ml/experiments/123');

    await waitFor(() => {
      expect(screen.getByText('this is logged models page')).toBeInTheDocument();
    });
  });
});
