import { describe, test, expect } from '@jest/globals';
import type { ICellRendererParams } from '@ag-grid-community/core';
import { render, screen, waitFor } from '../../../../../../common/utils/TestUtils.react18';
import { GroupParentCellRenderer } from './GroupParentCellRenderer';
import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../../../../common/utils/TestUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { testRoute, TestRouter } from '../../../../../../common/utils/RoutingTestUtils';
import { RunGroupingMode } from '../../../utils/experimentPage.row-types';

describe('GroupParentCellRenderer', () => {
  const dummyRendererApi: ICellRendererParams = {} as any;

  const renderComponent = (groupParentInfo: any) =>
    render(
      <GroupParentCellRenderer
        {...dummyRendererApi}
        data={
          {
            hidden: false,
            groupParentInfo,
          } as any
        }
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <MockedReduxStoreProvider state={{ entities: { colorByRunUuid: {} } }}>
                <TestRouter routes={[testRoute(children)]} />
              </MockedReduxStoreProvider>
            </DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );

  test('renders link with run IDs when groupingValues is empty', async () => {
    renderComponent({
      groupId: 'group-1',
      aggregatedMetricData: {},
      aggregatedParamData: {},
      groupingValues: [],
      isRemainingRunsGroup: false,
      runUuids: ['run-1', 'run-2'],
    });

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      expect(linkElement).toBeInTheDocument();

      const href = linkElement.getAttribute('href');
      // Should fall back to run IDs when no grouping values
      expect(href).toContain('searchFilter=attributes.run_id+IN');
      expect(href).toContain('isPreview=true');
    });
  });

  test('renders link with run IDs for remaining runs group', async () => {
    renderComponent({
      groupId: 'param.model_type',
      aggregatedMetricData: {},
      aggregatedParamData: {},
      groupingValues: [],
      isRemainingRunsGroup: true,
      runUuids: ['run-1', 'run-2'],
    });

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      const href = linkElement.getAttribute('href');
      // Remaining runs group should use run IDs
      expect(href).toContain('searchFilter=attributes.run_id+IN');
    });
  });

  test('renders link with param filter for param-grouped runs', async () => {
    renderComponent({
      groupId: 'param.model_type.tree',
      aggregatedMetricData: {},
      aggregatedParamData: {},
      groupingValues: [
        {
          mode: RunGroupingMode.Param,
          groupByData: 'model_type',
          value: 'tree',
        },
      ],
      isRemainingRunsGroup: false,
      runUuids: ['run-1', 'run-2'],
    });

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      const href = linkElement.getAttribute('href');
      // Should contain params.model_type = 'tree' (URL encoded)
      // params.model_type = 'tree' -> params.model_type+%3D+%27tree%27
      expect(href).toContain('params.model_type');
      expect(href).not.toContain('attributes.run_id');
      expect(href).toContain('isPreview=true');
    });
  });

  test('renders link with tag filter for tag-grouped runs', async () => {
    renderComponent({
      groupId: 'tag.experiment_version.v1',
      aggregatedMetricData: {},
      aggregatedParamData: {},
      groupingValues: [
        {
          mode: RunGroupingMode.Tag,
          groupByData: 'experiment_version',
          value: 'v1',
        },
      ],
      isRemainingRunsGroup: false,
      runUuids: ['run-1', 'run-2'],
    });

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      const href = linkElement.getAttribute('href');
      // Should contain tags.experiment_version = 'v1'
      expect(href).toContain('tags.experiment_version');
      expect(href).not.toContain('attributes.run_id');
    });
  });

  test('renders link with multiple filters for multi-criteria grouping', async () => {
    renderComponent({
      groupId: 'param.alpha.0.5,tag.env.prod',
      aggregatedMetricData: {},
      aggregatedParamData: {},
      groupingValues: [
        { mode: RunGroupingMode.Param, groupByData: 'alpha', value: '0.5' },
        { mode: RunGroupingMode.Tag, groupByData: 'env', value: 'prod' },
      ],
      isRemainingRunsGroup: false,
      runUuids: ['run-1', 'run-2'],
    });

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      const href = linkElement.getAttribute('href');
      // Should contain both filters joined with AND
      expect(href).toContain('params.alpha');
      expect(href).toContain('tags.env');
      expect(href).toContain('AND');
      expect(href).not.toContain('attributes.run_id');
    });
  });

  test('renders link with dataset filter for dataset-grouped runs', async () => {
    renderComponent({
      groupId: 'dataset.dataset.training_data.abc123',
      aggregatedMetricData: {},
      aggregatedParamData: {},
      groupingValues: [
        {
          mode: RunGroupingMode.Dataset,
          groupByData: 'dataset',
          value: { name: 'training_data', digest: 'abc123' },
        },
      ],
      isRemainingRunsGroup: false,
      runUuids: ['run-1'],
    });

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      const href = linkElement.getAttribute('href');
      // Should contain dataset.name and dataset.digest
      expect(href).toContain('dataset.name');
      expect(href).toContain('dataset.digest');
      expect(href).not.toContain('attributes.run_id');
    });
  });
});
