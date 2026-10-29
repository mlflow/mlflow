import { describe, expect, jest, test } from '@jest/globals';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { createMLflowRoutePath, useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { ExperimentEntity } from '../../types';
import { ExperimentPageBreadcrumbs } from './ExperimentPageBreadcrumbs';
import {
  ExperimentPageRevampProvider,
  useSetExperimentBreadcrumbPairwiseReviewAction,
  useSetExperimentBreadcrumbTitle,
} from './ExperimentPageRevampContext';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

const experiment: ExperimentEntity = {
  experimentId: '123',
  name: '/Users/test/Helpful experiment',
  artifactLocation: 'file:///tmp/123',
  lifecycleStage: 'active',
  creationTime: 0,
  lastUpdateTime: 0,
  tags: [],
};

const BreadcrumbHarness = ({ title, status }: { title?: string; status?: string }) => {
  useSetExperimentBreadcrumbTitle(title, status);
  return <ExperimentPageBreadcrumbs experiment={experiment} />;
};

const RouteAwareBreadcrumbHarness = () => {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  useSetExperimentBreadcrumbTitle(pathname.endsWith('dataset-1') ? 'First dataset' : undefined);
  return (
    <>
      <ExperimentPageBreadcrumbs experiment={experiment} />
      <button onClick={() => navigate(createMLflowRoutePath('/experiments/123/datasets/dataset-2'))}>
        Open another dataset
      </button>
    </>
  );
};

const ReviewActionHarness = ({ action }: { action?: () => void }) => {
  const { pathname } = useLocation();
  // Publishes only on run-1's page, so another run's page must not pick up run-1's action.
  useSetExperimentBreadcrumbPairwiseReviewAction(pathname.includes('/run-1/') ? action : undefined);
  const navigate = useNavigate();
  return (
    <>
      <ExperimentPageBreadcrumbs experiment={experiment} />
      <button
        onClick={() =>
          navigate(
            createMLflowRoutePath('/experiments/123/evaluation-runs/run-9/details?compare=true&compareRuns=run-2'),
          )
        }
      >
        Open another run
      </button>
    </>
  );
};

describe('ExperimentPageBreadcrumbs', () => {
  const { history } = setupTestRouter();

  const renderAt = (entry: string, routePath: string, title?: string, routeAware = false, status?: string) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <TestRouter
            history={history}
            initialEntries={[createMLflowRoutePath(entry)]}
            routes={[
              testRoute(
                <ExperimentPageRevampProvider enabled>
                  {routeAware ? <RouteAwareBreadcrumbHarness /> : <BreadcrumbHarness title={title} status={status} />}
                </ExperimentPageRevampProvider>,
                createMLflowRoutePath(routePath),
              ),
            ]}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );

  test('shows experiment and active list tab with an accessible copy action', async () => {
    renderAt('/experiments/123/datasets', '/experiments/:experimentId/datasets');

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByRole('link', { name: 'Experiments' })).toBeInTheDocument();
    expect(within(breadcrumb).getByRole('link', { name: 'Helpful experiment' })).toBeInTheDocument();
    expect(within(breadcrumb).getByText('Datasets')).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: 'Copy link to this experiment' })).toBeInTheDocument();
  });

  test.each([
    {
      entry: '/experiments/123/datasets/dataset-1',
      route: '/experiments/:experimentId/datasets/:datasetId',
      tab: 'Datasets',
      leaf: 'Friendly dataset',
      title: 'Friendly dataset',
    },
    {
      entry: '/experiments/123/evaluation-runs/run-1/details',
      route: '/experiments/:experimentId/evaluation-runs/:runUuid/details',
      tab: 'Evaluation runs',
      leaf: 'Helpful run',
      title: 'Helpful run',
    },
    {
      entry: '/experiments/123/prompts/prompt-1',
      route: '/experiments/:experimentId/prompts/:promptName',
      tab: 'Prompts',
      leaf: 'prompt-1',
    },
    {
      entry: '/experiments/123/review-queue?selectedQueueId=queue-1',
      route: '/experiments/:experimentId/review-queue',
      tab: 'Review',
      leaf: 'Helpful queue',
      title: 'Helpful queue',
    },
  ])('shows $leaf under $tab', async ({ entry, route, tab, leaf, title }) => {
    renderAt(entry, route, title);

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByRole('link', { name: tab })).toBeInTheDocument();
    expect(await within(breadcrumb).findByText(leaf)).toHaveAttribute('aria-current', 'page');
  });

  test('shows the run status pill next to the breadcrumb, outside the nav landmark', async () => {
    renderAt(
      '/experiments/123/evaluation-runs/run-1/details',
      '/experiments/:experimentId/evaluation-runs/:runUuid/details',
      'Helpful run',
      false,
      'RUNNING',
    );

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(await within(breadcrumb).findByText('Helpful run')).toHaveAttribute('aria-current', 'page');
    // The pill hugs the crumb but renders outside the Breadcrumb navigation landmark.
    expect(screen.getByText('RUNNING')).toBeInTheDocument();
    expect(within(breadcrumb).queryByText('RUNNING')).not.toBeInTheDocument();
  });

  test('shows no status pill when no run status is published', async () => {
    renderAt(
      '/experiments/123/evaluation-runs/run-1/details',
      '/experiments/:experimentId/evaluation-runs/:runUuid/details',
      'Helpful run',
    );

    await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(screen.queryByText('RUNNING')).not.toBeInTheDocument();
  });

  test('keeps Traces as the deepest breadcrumb for trace details', async () => {
    renderAt('/experiments/123/traces/trace-1', '/experiments/:experimentId/traces/:traceId');

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByText('Traces')).toHaveAttribute('aria-current', 'page');
    expect(within(breadcrumb).queryByText('trace-1')).not.toBeInTheDocument();
  });

  test('uses the published title for the matching route only', async () => {
    renderAt('/experiments/123/datasets/dataset-1', '/experiments/:experimentId/datasets/:datasetId', undefined, true);

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(await within(breadcrumb).findByText('First dataset')).toHaveAttribute('aria-current', 'page');
  });

  test('keeps a session conversation under Traces, where sessions are grouped', async () => {
    renderAt('/experiments/123/chat-sessions/session-1', '/experiments/:experimentId/chat-sessions/:sessionId');

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByRole('link', { name: 'Traces' })).toBeInTheDocument();
    expect(within(breadcrumb).getByText('session-1')).toHaveAttribute('aria-current', 'page');
  });

  test.each([
    { name: 'experiment', target: 'Helpful experiment' },
    { name: 'leaf', target: 'Friendly dataset' },
  ])('shows the full $name name in a tooltip', async ({ target }) => {
    const user = userEvent.setup();
    renderAt(
      '/experiments/123/datasets/dataset-1',
      '/experiments/:experimentId/datasets/:datasetId',
      'Friendly dataset',
    );

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    await user.hover(within(breadcrumb).getByText(target));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(target);
  });

  test('does not show a title published for a previous route', async () => {
    renderAt('/experiments/123/datasets/dataset-1', '/experiments/:experimentId/datasets/:datasetId', undefined, true);
    expect(await screen.findByText('First dataset')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Open another dataset' }));

    expect(await screen.findByText('dataset-2')).toBeInTheDocument();
    expect(screen.queryByText('First dataset')).not.toBeInTheDocument();
  });
  test.each([
    { flag: true, entry: '/experiments/123/evaluation-runs/run-1/details?compare=true&compareRuns=run-2', shown: true },
    { flag: true, entry: '/experiments/123/evaluation-runs/run-1/details?compareRuns=run-2', shown: false },
    {
      flag: false,
      entry: '/experiments/123/evaluation-runs/run-1/details?compare=true&compareRuns=run-2',
      shown: false,
    },
  ])('shows the exit-comparison action only while comparing on the run page (%o)', async ({ flag, entry, shown }) => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(flag);
    renderAt(entry, '/experiments/:experimentId/evaluation-runs/:runUuid/details', 'Run one');

    await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(Boolean(screen.queryByRole('button', { name: 'Back to Reference Run' }))).toBe(shown);
  });

  test('keeps the exit-comparison action off other pages', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderAt('/experiments/123/datasets?compare=true', '/experiments/:experimentId/datasets');

    await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(screen.queryByRole('button', { name: 'Back to Reference Run' })).not.toBeInTheDocument();
  });

  const renderWithReviewAction = (entry: string, action?: () => void, enabled = true) =>
    render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <TestRouter
            history={history}
            initialEntries={[createMLflowRoutePath(entry)]}
            routes={[
              testRoute(
                <ExperimentPageRevampProvider enabled={enabled}>
                  <ReviewActionHarness action={action} />
                </ExperimentPageRevampProvider>,
                createMLflowRoutePath('/experiments/:experimentId/evaluation-runs/:runUuid/details'),
              ),
            ]}
          />
        </DesignSystemProvider>
      </IntlProvider>,
    );
  const COMPARE_ENTRY = '/experiments/123/evaluation-runs/run-1/details?compare=true&compareRuns=run-2';

  test('shows Review next to Back to Reference Run while a review action is published, and runs it', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const onReview = jest.fn();
    renderWithReviewAction(COMPARE_ENTRY, onReview);

    const review = await screen.findByRole('button', { name: 'Review' });
    expect(review).toHaveAttribute('aria-keyshortcuts', 'R');
    expect(screen.getByRole('button', { name: 'Back to Reference Run' }).parentElement).toBe(review.parentElement);
    await userEvent.click(review);
    expect(onReview).toHaveBeenCalledTimes(1);
  });

  test('shows no Review without a published action, outside comparisons, or with the shell disabled', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const { unmount } = renderWithReviewAction(COMPARE_ENTRY, undefined);
    await screen.findByRole('button', { name: 'Back to Reference Run' });
    expect(screen.queryByRole('button', { name: 'Review' })).not.toBeInTheDocument();
    unmount();

    const second = renderWithReviewAction('/experiments/123/evaluation-runs/run-1/details', jest.fn());
    await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(screen.queryByRole('button', { name: 'Review' })).not.toBeInTheDocument();
    second.unmount();

    renderWithReviewAction(COMPARE_ENTRY, jest.fn(), false);
    expect(screen.queryByRole('button', { name: 'Review' })).not.toBeInTheDocument();
  });

  test('drops a review action published for a previous route', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderWithReviewAction(COMPARE_ENTRY, jest.fn());
    await screen.findByRole('button', { name: 'Review' });
    await userEvent.click(screen.getByRole('button', { name: 'Open another run' }));
    // Still comparing on run-9, so Back to Reference Run stays; run-1's Review doesn't follow.
    await screen.findByRole('button', { name: 'Back to Reference Run' });
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Review' })).not.toBeInTheDocument());
  });

  test('shows no Review with the flag off, even with a published action', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    renderWithReviewAction(COMPARE_ENTRY, jest.fn());
    await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(screen.queryByRole('button', { name: 'Review' })).not.toBeInTheDocument();
  });
});
