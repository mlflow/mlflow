import { describe, expect, test } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { createMLflowRoutePath, useLocation, useNavigate } from '../../../common/utils/RoutingUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { ExperimentEntity } from '../../types';
import { ExperimentPageBreadcrumbs } from './ExperimentPageBreadcrumbs';
import { ExperimentPageRevampProvider, useSetExperimentBreadcrumbTitle } from './ExperimentPageRevampContext';

const experiment: ExperimentEntity = {
  experimentId: '123',
  name: '/Users/test/Helpful experiment',
  artifactLocation: 'file:///experiments/123',
  lifecycleStage: 'active',
  creationTime: 0,
  lastUpdateTime: 0,
  tags: [],
};

const BreadcrumbHarness = ({ title }: { title?: string }) => {
  useSetExperimentBreadcrumbTitle(title);
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
        Next dataset
      </button>
    </>
  );
};

describe('ExperimentPageBreadcrumbs', () => {
  const { history } = setupTestRouter();

  const renderAt = (entry: string, routePath: string, title?: string, routeAware = false) =>
    render(<div />, {
      wrapper: () => (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <TestRouter
              history={history}
              initialEntries={[createMLflowRoutePath(entry)]}
              routes={[
                testRoute(
                  <ExperimentPageRevampProvider enabled>
                    {routeAware ? <RouteAwareBreadcrumbHarness /> : <BreadcrumbHarness title={title} />}
                  </ExperimentPageRevampProvider>,
                  createMLflowRoutePath(routePath),
                ),
                testRoute(
                  <div>Experiment settings destination</div>,
                  createMLflowRoutePath('/experiments/:experimentId/settings'),
                ),
              ]}
            />
          </DesignSystemProvider>
        </IntlProvider>
      ),
    });

  test('shows the experiment and active tab with an accessible copy action', async () => {
    renderAt('/experiments/123/datasets', '/experiments/:experimentId/datasets');

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByRole('link', { name: 'Experiments' })).toBeInTheDocument();
    expect(within(breadcrumb).getByRole('link', { name: 'Helpful experiment' })).toBeInTheDocument();
    expect(within(breadcrumb).getByText('Datasets')).toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: 'Experiment Settings' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Copy link to this experiment' })).toBeInTheDocument();
  });

  test('opens experiment settings from the gear action', async () => {
    renderAt('/experiments/123/datasets', '/experiments/:experimentId/datasets');

    await userEvent.click(await screen.findByRole('button', { name: 'Experiment Settings' }));

    expect(await screen.findByText('Experiment settings destination')).toBeInTheDocument();
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
      entry: '/experiments/123/chat-sessions/session-1',
      route: '/experiments/:experimentId/chat-sessions/:sessionId',
      tab: 'Chat sessions',
      leaf: 'session-1',
    },
    {
      entry: '/experiments/123/prompts/prompt-1',
      route: '/experiments/:experimentId/prompts/:promptName',
      tab: 'Prompts',
      leaf: 'prompt-1',
    },
  ])('shows $leaf under $tab', async ({ entry, route, tab, leaf, title }) => {
    renderAt(entry, route, title);

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByRole('link', { name: tab })).toBeInTheDocument();
    expect(await within(breadcrumb).findByText(leaf)).toHaveAttribute('aria-current', 'page');
  });

  test('keeps Traces as the deepest breadcrumb for trace details', async () => {
    renderAt('/experiments/123/traces/trace-1', '/experiments/:experimentId/traces/:traceId');

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(within(breadcrumb).getByText('Traces')).toHaveAttribute('aria-current', 'page');
    expect(within(breadcrumb).queryByText('trace-1')).not.toBeInTheDocument();
  });

  test('does not show a title published for a previous route', async () => {
    renderAt('/experiments/123/datasets/dataset-1', '/experiments/:experimentId/datasets/:datasetId', undefined, true);
    expect(await screen.findByText('First dataset')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Next dataset' }));

    expect(await screen.findByText('dataset-2')).toBeInTheDocument();
    expect(screen.queryByText('First dataset')).not.toBeInTheDocument();
  });
});
