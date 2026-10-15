import { describe, expect, it } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '../../../common/utils/RoutingTestUtils';

import { RunDetailsSidebar, type RunDetailsScorer, type RunDetailsSidebarProps } from './RunDetailsSidebar';

const tags = Array.from({ length: 12 }, (_, index) => ({
  key: `demo.tag_${index}`,
  value: `value-${index}`,
}));
const longTag = {
  key: 'demo.long_tag',
  value: 'a-very-long-unbroken-value-that-needs-to-be-truncated-within-the-sidebar',
};
const createScorer = (overrides: Partial<RunDetailsScorer> = {}): RunDetailsScorer => ({
  name: 'quality',
  displayName: 'quality',
  ...overrides,
});

describe('RunDetailsSidebar layout', () => {
  const { history } = setupTestRouter();

  const renderSidebar = async (props: Partial<RunDetailsSidebarProps> = {}) => {
    render(
      <RunDetailsSidebar
        experimentId="exp-123"
        stats={{ traces: 7 }}
        scorers={[]}
        tags={[...tags, longTag]}
        {...props}
      />,
      {
        wrapper: ({ children }) => (
          <TestRouter
            history={history}
            initialEntries={['/experiments/exp-123/evaluation-runs/run-123/details']}
            routes={[
              testRoute(
                <IntlProvider locale="en">
                  <DesignSystemProvider>{children}</DesignSystemProvider>
                </IntlProvider>,
                '/experiments/:experimentId/evaluation-runs/:runUuid/details',
              ),
            ]}
          />
        ),
      },
    );
    await waitForRoutesToBeRendered();
  };

  it('does not display an unknown trace count as zero', async () => {
    await renderSidebar({ stats: {} });

    expect(screen.queryByText('0')).not.toBeInTheDocument();
    // Includes the Dataset and Dataset source type rows, which show a dash when no dataset is present.
    expect(screen.getAllByText('—')).toHaveLength(6);
  });

  it('preserves zero usage and displays costs below one cent', async () => {
    await renderSidebar({ stats: { traces: 4, totalTokens: 0, totalCost: 0.000123, errors: 0 } });

    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getAllByText('0')).toHaveLength(2);
    expect(screen.getByText('$0.000123')).toBeInTheDocument();
  });

  it('uses a constrained, right-aligned flex column', async () => {
    await renderSidebar();

    expect(screen.getByRole('complementary')).toHaveStyle({
      flex: '0 1 320px',
      marginLeft: 'auto',
      minWidth: 'min(260px, 100%)',
      maxWidth: '100%',
    });
  });

  it('shows the complete tag text in a tooltip when the displayed text is truncated', async () => {
    await renderSidebar();
    const user = userEvent.setup();

    const sidebar = within(screen.getByRole('complementary'));
    expect(sidebar.getByText('demo.tag_0: value-0')).toBeInTheDocument();
    expect(sidebar.getByText('demo.tag_11: value-11')).toBeInTheDocument();

    const longTagText = `${longTag.key}: ${longTag.value}`;
    const longTagContent = sidebar.getByText(longTagText);
    expect(longTagContent).toHaveStyle(`
      min-width: 0;
      max-width: 100%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    `);
    Object.defineProperties(longTagContent, {
      clientWidth: { configurable: true, value: 100 },
      scrollWidth: { configurable: true, value: 400 },
    });

    await user.hover(longTagContent);

    expect(await screen.findByRole('tooltip')).toHaveTextContent(longTagText);
  });

  it('does not show a tooltip when the complete tag text fits', async () => {
    await renderSidebar({ tags: [tags[0]] });
    const user = userEvent.setup();
    const tagContent = screen.getByText('demo.tag_0: value-0');
    Object.defineProperties(tagContent, {
      clientWidth: { configurable: true, value: 100 },
      scrollWidth: { configurable: true, value: 100 },
    });

    await user.hover(tagContent);

    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it.each([
    [
      'creator',
      { createdBy: 'creator-with-an-extremely-long-unbroken-identity@example.com' },
      'creator-with-an-extremely-long-unbroken-identity@example.com',
    ],
    [
      'scorer name',
      {
        scorers: [
          createScorer({
            name: 'long-scorer',
            displayName: 'scorer-with-an-extremely-long-unbroken-name',
          }),
        ],
      },
      'scorer-with-an-extremely-long-unbroken-name',
    ],
  ])(
    'preserves the complete %s in its title when the displayed value is constrained',
    async (_field, props, longValue) => {
      await renderSidebar(props);

      expect(screen.getByTitle(longValue)).toHaveStyle(`
        min-width: 0;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      `);
    },
  );

  it('shows the dataset name and source type', async () => {
    await renderSidebar({ dataset: { name: 'eval-set', sourceType: 'code', digest: 'abc123' } });

    expect(screen.getByText('eval-set')).toBeInTheDocument();
    expect(screen.getByText('code')).toBeInTheDocument();
  });
});
