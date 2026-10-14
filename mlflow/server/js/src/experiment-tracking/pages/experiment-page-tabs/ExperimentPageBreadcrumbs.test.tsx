import { describe, expect, test } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { createMLflowRoutePath, useLocation } from '../../../common/utils/RoutingUtils';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { ExperimentEntity } from '../../types';
import { ExperimentPageBreadcrumbs } from './ExperimentPageBreadcrumbs';
import { ExperimentPageRevampProvider, useSetExperimentBreadcrumbTitle } from './ExperimentPageRevampContext';

const experiment: ExperimentEntity = {
  experimentId: '123',
  name: '/Users/test/Helpful experiment',
  artifactLocation: 'file:///tmp/123',
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
  useSetExperimentBreadcrumbTitle(pathname.endsWith('dataset-1') ? 'First dataset' : undefined);
  return <ExperimentPageBreadcrumbs experiment={experiment} />;
};

describe('ExperimentPageBreadcrumbs', () => {
  const { history } = setupTestRouter();

  const renderAt = (entry: string, routePath: string, title?: string, routeAware = false) =>
    render(
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

  test('uses the published title for the matching route only', async () => {
    renderAt('/experiments/123/datasets/dataset-1', '/experiments/:experimentId/datasets/:datasetId', undefined, true);

    const breadcrumb = await screen.findByRole('navigation', { name: 'Breadcrumb' });
    expect(await within(breadcrumb).findByText('First dataset')).toHaveAttribute('aria-current', 'page');
  });
});
