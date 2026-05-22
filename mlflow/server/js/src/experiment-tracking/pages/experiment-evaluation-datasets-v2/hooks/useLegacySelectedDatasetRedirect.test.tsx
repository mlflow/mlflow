// @ts-nocheck — punting test typing; see PR2 plan in branch import { afterEach, describe, expect, jest, test } from '@jest/globals';
import { render, screen, waitFor } from '@testing-library/react';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '@mlflow/mlflow/src/common/utils/RoutingTestUtils';
import { useLegacySelectedDatasetRedirect } from './useLegacySelectedDatasetRedirect';

const Probe = () => {
  const { isRedirecting } = useLegacySelectedDatasetRedirect();
  return <div data-testid="probe">{isRedirecting ? 'redirecting' : 'idle'}</div>;
};

const TargetStub = () => <div data-testid="detail-target">detail</div>;

describe('useLegacySelectedDatasetRedirect', () => {
  // setupTestRouter registers beforeAll/afterAll hooks; must live at describe scope.
  const { history } = setupTestRouter();

  const renderWithRouter = (initialUrl: string) => {
    const result = render(
      <TestRouter
        routes={[
          testRoute(<Probe />, '/experiments/:experimentId/datasets'),
          testRoute(<TargetStub />, '/experiments/:experimentId/datasets/:datasetId'),
        ]}
        history={history}
        initialEntries={[initialUrl]}
      />,
    );
    return { ...result, history };
  };

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('rewrites a legacy `?selectedDatasetId=…` URL to the V2 detail route', async () => {
    const { history } = renderWithRouter('/experiments/exp-1/datasets?selectedDatasetId=ds-42');

    await waitFor(() => {
      expect(history.location.pathname).toContain('/experiments/exp-1/datasets/ds-42');
    });
    expect(history.location.search).not.toContain('selectedDatasetId');
  });

  test('preserves unrelated query params when redirecting', async () => {
    const { history } = renderWithRouter('/experiments/exp-1/datasets?selectedDatasetId=ds-42&keep=me&another=value');

    await waitFor(() => {
      expect(history.location.pathname).toContain('/experiments/exp-1/datasets/ds-42');
    });
    expect(history.location.search).toContain('keep=me');
    expect(history.location.search).toContain('another=value');
    expect(history.location.search).not.toContain('selectedDatasetId');
  });

  test('is a no-op when the legacy param is absent', async () => {
    const { history } = renderWithRouter('/experiments/exp-1/datasets');

    await waitForRoutesToBeRendered();
    const probe = await screen.findByTestId('probe');
    expect(probe.textContent).toBe('idle');
    // Stayed on the list URL — pathname ends in `/datasets`, not `/datasets/<id>`.
    expect(history.location.pathname).toMatch(/\/experiments\/exp-1\/datasets$/);
  });
});