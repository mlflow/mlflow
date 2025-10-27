import type { ICellRendererParams } from '@ag-grid-community/core';
import { render, screen, waitFor } from '../../../../../../common/utils/TestUtils.react18';
import { GroupParentCellRenderer } from './GroupParentCellRenderer';
import { IntlProvider } from 'react-intl';
import { MockedReduxStoreProvider } from '../../../../../../common/utils/TestUtils';
import { DesignSystemProvider } from '@databricks/design-system';
import { testRoute, TestRouter } from '../../../../../../common/utils/RoutingTestUtils';

describe('GroupParentCellRenderer', () => {
  const dummyRendererApi: ICellRendererParams = {} as any;

  test('renders link to runs in the group', async () => {
    render(
      <GroupParentCellRenderer
        {...dummyRendererApi}
        data={
          {
            hidden: false,
            groupParentInfo: {
              groupId: 'group-1',
              aggregatedMetricData: {},
              aggregatedParamData: {},
              groupingValues: [],
              isRemainingRunsGroup: false,
              runUuids: ['run-1, run-2'],
            },
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

    await waitFor(() => {
      const linkElement = screen.getByRole('link', { hidden: true });
      expect(linkElement).toBeInTheDocument();

      const href = linkElement.getAttribute('href');
      expect(href).toContain('searchFilter=attributes.run_id+IN+%28%27run-1%2C+run-2%27%29');
      expect(href).toContain('isPreview=true');
    });
  });
});
