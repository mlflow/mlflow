import { beforeEach, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';

import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG } from '../../constants';
import { useIssueDetectionSettings } from './useIssueDetectionSettings';

const tagWith = (instructions: string) => [
  { key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG, value: JSON.stringify({ instructions, categories: {} }) },
];

/** Stored tag the server answers reads with. Reads resolve only once released. */
let storedInstructions: string;
let holdReads: boolean;
let releaseReads: () => void;
let readsReleased: Promise<void>;

const server = setupServer(
  rest.get('*/ajax-api/2.0/mlflow/experiments/get', async (_req, res, ctx) => {
    // Captured before awaiting, so a held read answers with the tag as it was when it started
    // -- which is the whole point: it is stale by the time it lands.
    const answer = storedInstructions;
    if (holdReads) {
      await readsReleased;
    }
    return res(ctx.json({ experiment: { experiment_id: '1', name: 'exp', tags: tagWith(answer) } }));
  }),
  rest.post('*/ajax-api/2.0/mlflow/experiments/set-experiment-tag', async (req, res, ctx) => {
    const body = (await req.json()) as { value: string };
    storedInstructions = JSON.parse(body.value).instructions;
    return res(ctx.json({}));
  }),
);

/** Surfaces the hook's state and actions so the race can be driven step by step. */
const Harness = ({ onReady }: { onReady: (api: ReturnType<typeof useIssueDetectionSettings>) => void }) => {
  const api = useIssueDetectionSettings('1');
  onReady(api);
  return <div data-testid="instructions">{api.data?.instructions ?? '(none)'}</div>;
};

beforeEach(() => {
  storedInstructions = 'Old priorities.';
  holdReads = false;
  readsReleased = new Promise<void>((resolve) => {
    releaseReads = resolve;
  });
});

describe('useIssueDetectionSettings', () => {
  test('a read still in flight when a save lands does not resurrect the old settings', async () => {
    // The read was issued before the write, so it answers with the settings as they were. If
    // it lands after the cache update, the next detection run is submitted with the priorities
    // the user had just replaced -- which is invisible, because the tag is correct.
    let api!: ReturnType<typeof useIssueDetectionSettings>;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderWithDesignSystem(
      <QueryClientProvider client={client}>
        <Harness
          onReady={(next) => {
            api = next;
          }}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(screen.getByTestId('instructions')).toHaveTextContent('Old priorities.'));

    // A refetch that will answer with the pre-save tag, held open across the save.
    holdReads = true;
    void api.refetch();
    await waitFor(() => expect(api.isFetching).toBe(true));

    await api.save.mutateAsync({ instructions: 'New priorities.', categories: {} });
    await waitFor(() => expect(screen.getByTestId('instructions')).toHaveTextContent('New priorities.'));

    // Now let the stale read answer. This is the assertion that matters: without the save
    // cancelling it first, its pre-save payload lands here and wins.
    releaseReads();
    await waitFor(() => expect(api.isFetching).toBe(false));

    expect(screen.getByTestId('instructions')).toHaveTextContent('New priorities.');
  });

  test('an ordinary read after a save still reaches the server', async () => {
    // Cancelling on save must not leave the query permanently stuck on cached data.
    let api!: ReturnType<typeof useIssueDetectionSettings>;
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    renderWithDesignSystem(
      <QueryClientProvider client={client}>
        <Harness
          onReady={(next) => {
            api = next;
          }}
        />
      </QueryClientProvider>,
    );

    await waitFor(() => expect(screen.getByTestId('instructions')).toHaveTextContent('Old priorities.'));
    await api.save.mutateAsync({ instructions: 'Saved here.', categories: {} });

    storedInstructions = 'Changed elsewhere.';
    await api.refetch();

    await waitFor(() => expect(screen.getByTestId('instructions')).toHaveTextContent('Changed elsewhere.'));
  });
});
