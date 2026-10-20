import { describe, expect, it } from '@jest/globals';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import { rest } from 'msw';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../../common/utils/setup-msw';
import {
  setupTestRouter,
  testRoute,
  TestRouter,
  waitForRoutesToBeRendered,
} from '../../../common/utils/RoutingTestUtils';

import Routes from '../../routes';
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
    // Includes the Dataset, Dataset digest, Dataset source type and Prompt rows, which show a dash when the run
    // has no dataset or linked prompt.
    expect(screen.getAllByText('—')).toHaveLength(8);
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

  it('shows the dataset digest and labels a default-named dataset as unnamed', async () => {
    await renderSidebar({ dataset: { name: 'dataset', sourceType: 'code', digest: 'abc123' } });

    expect(screen.getByText('abc123')).toBeInTheDocument();
    expect(screen.getByText('Unnamed dataset')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /View dataset/ })).not.toBeInTheDocument();
  });

  it('links a dataset logged from an evaluation dataset to its page', async () => {
    await renderSidebar({
      dataset: {
        name: 'golden-set',
        sourceType: 'mlflow_evaluation_dataset',
        digest: 'abc123',
        source: JSON.stringify({ dataset_id: 'd-123' }),
      },
    });

    const link = screen.getByRole('link', { name: 'View dataset golden-set' });
    expect(link).toHaveAttribute('href', Routes.getExperimentPageDatasetDetailRoute('exp-123', 'd-123'));
    expect(link).toHaveAttribute('target', '_blank');
  });

  it('links each linked prompt to its exact version', async () => {
    await renderSidebar({
      prompts: [
        { name: 'support prompt', version: '2' },
        { name: 'router', version: '5' },
      ],
    });

    const link = screen.getByRole('link', { name: 'View prompt support prompt version 2' });
    expect(link).toHaveTextContent('support prompt (v2)');
    expect(link).toHaveAttribute(
      'href',
      `${Routes.getPromptDetailsPageRoute(encodeURIComponent('support prompt'), 'exp-123')}?promptVersion=2`,
    );
    expect(link).toHaveAttribute('target', '_blank');
    expect(screen.getByRole('link', { name: 'View prompt router version 5' })).toBeInTheDocument();
  });

  it('links a scorer to the registered scorer that produced its assessments', async () => {
    await renderSidebar({
      scorers: [
        createScorer({ name: 'quality', displayName: 'Quality', scorerReference: { scorerName: 'quality/v2' } }),
        createScorer({ name: 'tone', displayName: 'Tone' }),
      ],
    });

    const link = screen.getByRole('link', { name: 'View scorer Quality' });
    expect(link).toHaveAttribute('href', '/experiments/exp-123/judges?selectedScorerName=quality%2Fv2');
    expect(link).toHaveAttribute('target', '_blank');
    expect(screen.getByText('Tone')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'View scorer Tone' })).not.toBeInTheDocument();
  });
});

describe('RunDetailsSidebar evaluation lineage', () => {
  const { history } = setupTestRouter();
  const server = setupServer();

  const renderSidebar = async (props: Partial<RunDetailsSidebarProps> = {}) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(<RunDetailsSidebar experimentId="exp-123" stats={{}} scorers={[]} tags={[]} {...props} />, {
      wrapper: ({ children }) => (
        <TestRouter
          history={history}
          initialEntries={['/experiments/exp-123/evaluation-runs/run-123/details']}
          routes={[
            testRoute(
              <QueryClientProvider client={client}>
                <IntlProvider locale="en">
                  <DesignSystemProvider>{children}</DesignSystemProvider>
                </IntlProvider>
              </QueryClientProvider>,
              '/experiments/:experimentId/evaluation-runs/:runUuid/details',
            ),
          ]}
        />
      ),
    });
    await waitForRoutesToBeRendered();
  };

  it('shows no lineage rows when the run has no lineage', async () => {
    await renderSidebar({ scorers: [createScorer()] });

    expect(screen.queryByText('Agent')).not.toBeInTheDocument();
    expect(screen.queryByText('Git commit')).not.toBeInTheDocument();
    expect(screen.queryByText(/^[0-9a-f]{7}$/)).not.toBeInTheDocument();
  });

  it('shows the scorers digest only in the Scorers heading tooltip', async () => {
    const user = userEvent.setup();
    await renderSidebar({
      scorers: [createScorer()],
      scorersDigest: { digest: 'sha256:complete', hashed: 2, total: 2, coverage: 'complete' },
    });

    expect(screen.queryByText(/sha256:complete/)).not.toBeInTheDocument();
    await user.hover(screen.getByText('Scorers (1)'));
    const tooltip = await screen.findByRole('tooltip');
    expect(tooltip).toHaveTextContent('Scorers digest: sha256:complete');
    expect(tooltip).toHaveTextContent('A fingerprint of all scorers in this run.');
  });

  it('puts the coverage of a partial scorers digest in the tooltip', async () => {
    const user = userEvent.setup();
    await renderSidebar({
      scorers: [createScorer()],
      scorersDigest: { digest: 'sha256:partial', hashed: 1, total: 3, coverage: 'partial' },
    });

    await user.hover(screen.getByText('Scorers (1)'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      'A fingerprint of 1 of 3 scorers in this run. MLflow could not read the others.',
    );
  });

  it('explains an unavailable scorers digest in the tooltip', async () => {
    const user = userEvent.setup();
    await renderSidebar({
      scorers: [createScorer()],
      scorersDigest: { hashed: 0, total: 2, coverage: 'unavailable' },
    });

    await user.hover(screen.getByText('Scorers (1)'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent(
      'No scorers digest. MLflow could not read any scorer in this run.',
    );
  });

  it('shows a short definition digest next to each scorer that has one, and keeps the scorer link', async () => {
    const user = userEvent.setup();
    await renderSidebar({
      scorers: [
        createScorer({
          name: 'quality',
          displayName: 'quality',
          digest: 'sha256:3f9a1c2d4e5f',
          scorerReference: { scorerName: 'quality' },
        }),
        createScorer({ name: 'safety', displayName: 'safety' }),
      ],
    });

    expect(screen.getByRole('link', { name: 'View scorer quality' })).toBeInTheDocument();
    expect(screen.getByText('safety')).toBeInTheDocument();
    expect(screen.getAllByText(/^[0-9a-f]{7}$/)).toHaveLength(1);
    await user.hover(screen.getByText('3f9a1c2'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('sha256:3f9a1c2d4e5f');
  });

  it('shows the agent lineage and links the logged model', async () => {
    const user = userEvent.setup();
    server.use(
      rest.get('*/ajax-api/2.0/mlflow/logged-models/m-123', (_req, res, ctx) =>
        res(ctx.json({ model: { info: { model_id: 'm-123', name: 'support-agent-model' } } })),
      ),
    );
    await renderSidebar({
      agent: {
        function: 'my_pkg.agent.predict',
        uri: 'endpoints:/support-agent',
        digest: 'sha256:agent',
        loggedModelId: 'm-123',
      },
      gitCommit: '0123456789abcdef',
    });

    expect(screen.getByText('Agent')).toBeInTheDocument();
    expect(screen.getByText('endpoints:/support-agent')).toBeInTheDocument();
    expect(screen.queryByText('Source digest')).not.toBeInTheDocument();
    await user.hover(screen.getByText('my_pkg.agent.predict'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('my_pkg.agent.predict (source digest: sha256:agent)');
    expect(screen.getByTitle('0123456789abcdef')).toHaveTextContent('0123456');
    const modelLink = await screen.findByRole('link', { name: 'View logged model support-agent-model' });
    expect(modelLink).toHaveAttribute('href', expect.stringContaining('m-123'));
    expect(modelLink).toHaveAttribute('target', '_blank');
  });

  it('shows a short source digest when the agent has no function', async () => {
    const user = userEvent.setup();
    await renderSidebar({ agent: { digest: 'sha256:0123456789abcdef' } });

    expect(screen.getByText('Source digest')).toBeInTheDocument();
    await user.hover(screen.getByText('0123456'));
    expect(await screen.findByRole('tooltip')).toHaveTextContent('sha256:0123456789abcdef');
  });

  it('shows the git commit in Run Details for a run without agent lineage', async () => {
    await renderSidebar({ gitCommit: 'fedcba9876543210' });

    expect(screen.getByText('Git commit')).toBeInTheDocument();
    expect(screen.getByTitle('fedcba9876543210')).toHaveTextContent('fedcba9');
    expect(screen.queryByText('Agent')).not.toBeInTheDocument();
  });
});
