import { beforeEach, describe, expect, test } from '@jest/globals';
import { rest } from 'msw';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { setupServer } from '../../../common/utils/setup-msw';
import { QueryClient, QueryClientProvider } from '../../../common/utils/reactQueryHooks';
import { MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG } from '../../constants';
import { IssueDetectionSettingsControl } from './IssueDetectionSettingsControl';

interface SavedTag {
  key: string;
  value: string;
}

let experimentTags: SavedTag[];
let written: SavedTag[];
let saveFails: boolean;
/** Set to fail the experiment read, which is how a failed settings load is exercised. */
let loadFails: boolean;

const server = setupServer(
  rest.get('*/ajax-api/2.0/mlflow/experiments/get', (_req, res, ctx) =>
    loadFails
      ? res(ctx.status(500), ctx.json({ message: 'nope' }))
      : res(ctx.json({ experiment: { experiment_id: '1', name: 'exp', tags: experimentTags } })),
  ),
  rest.post('*/ajax-api/2.0/mlflow/experiments/set-experiment-tag', async (req, res, ctx) => {
    if (saveFails) {
      return res(ctx.status(500), ctx.json({ message: 'nope' }));
    }
    const body = (await req.json()) as { key: string; value: string };
    written.push({ key: body.key, value: body.value });
    return res(ctx.json({}));
  }),
);

const render = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, cacheTime: 0 } } });
  return renderWithDesignSystem(
    <QueryClientProvider client={client}>
      <IssueDetectionSettingsControl experimentId="1" />
    </QueryClientProvider>,
  );
};

const openSettings = async () => {
  const open = await screen.findByRole('button', { name: 'Issue detection settings' });
  await waitFor(() => expect(open).not.toBeDisabled());
  await userEvent.click(open);
  return screen.findByRole('textbox', { name: /Detection priorities/ });
};

/** The one tag the control writes, parsed. */
const savedSettings = () => JSON.parse(written.at(-1)!.value);

beforeEach(() => {
  experimentTags = [];
  written = [];
  saveFails = false;
  loadFails = false;
});

describe('IssueDetectionSettingsControl', () => {
  test('an experiment with no saved settings opens with every focus area selected', async () => {
    render();
    await openSettings();

    expect(screen.getByRole('button', { name: 'Correctness', pressed: true })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cost', pressed: true })).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Detection priorities/ })).toHaveValue('');
  });

  test('saved settings are shown, not the defaults', async () => {
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG,
        value: JSON.stringify({
          instructions: 'Only report high severity.',
          categories: { correctness: true, cost: false },
        }),
      },
    ];
    render();
    await openSettings();

    expect(screen.getByRole('textbox', { name: /Detection priorities/ })).toHaveValue('Only report high severity.');
    expect(screen.getByRole('button', { name: 'Cost', pressed: false })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Correctness', pressed: true })).toBeInTheDocument();
  });

  test('a focus area the saved settings never mentioned shows unticked', async () => {
    // Only what the tag selects is ticked, so a built-in added to the product after the owner
    // last saved shows off here until they save again.
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG,
        value: JSON.stringify({ instructions: '', categories: { correctness: true } }),
      },
    ];
    render();
    await openSettings();

    expect(screen.getByRole('button', { name: 'Correctness', pressed: true })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Cost', pressed: false })).toBeInTheDocument();
  });

  test('priorities and focus areas are saved as one tag', async () => {
    render();
    const priorities = await openSettings();

    await userEvent.type(priorities, 'Treat auth failures as high.');
    await userEvent.click(screen.getByRole('button', { name: 'Cost' }));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(written).toHaveLength(1));
    expect(written[0].key).toBe(MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG);
    expect(savedSettings().instructions).toBe('Treat auth failures as high.');
    expect(savedSettings().categories.cost).toBe(false);
    expect(savedSettings().categories.correctness).toBe(true);
  });

  test('priorities are trimmed before being stored', async () => {
    render();
    const priorities = await openSettings();

    await userEvent.type(priorities, '   Be strict about safety.   ');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(written).toHaveLength(1));
    expect(savedSettings().instructions).toBe('Be strict about safety.');
  });

  test('cancelling writes nothing and discards the draft', async () => {
    render();
    const priorities = await openSettings();

    await userEvent.type(priorities, 'Typed then abandoned.');
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));

    await waitFor(() =>
      expect(screen.queryByRole('textbox', { name: /Detection priorities/ })).not.toBeInTheDocument(),
    );
    expect(written).toHaveLength(0);

    // Reopening shows what is saved, not the abandoned draft.
    await openSettings();
    expect(screen.getByRole('textbox', { name: /Detection priorities/ })).toHaveValue('');
  });

  test('a custom focus area can be added and is selected', async () => {
    render();
    await openSettings();

    await userEvent.click(screen.getByRole('button', { name: 'Custom category' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Custom category' }), 'pii leakage');
    await userEvent.click(screen.getByRole('button', { name: 'Add' }));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(written).toHaveLength(1));
    expect(savedSettings().categories['pii leakage']).toBe(true);
  });

  test('a custom focus area that duplicates an existing one cannot be added', async () => {
    render();
    await openSettings();

    await userEvent.click(screen.getByRole('button', { name: 'Custom category' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Custom category' }), 'Correctness');

    // Disabled with no message, matching the managed product rather than explaining it.
    await waitFor(() => expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled());
    expect(screen.queryByText(/already exists/)).not.toBeInTheDocument();
  });

  test('a focus area name containing a template variable is rejected', async () => {
    // Reported in review: such a name saves fine but then reaches the judge's instructions,
    // where a doubled brace declares a template variable. The server rewrites it rather than
    // failing, so the name it judges against would no longer be the name stored here.
    render();
    await openSettings();

    await userEvent.click(screen.getByRole('button', { name: 'Custom category' }));
    await userEvent.type(screen.getByRole('textbox', { name: 'Custom category' }), 'Unresolved {{ name }}');

    expect(await screen.findByText('A category name cannot contain {{ }}.')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add' })).toBeDisabled();
  });

  test('a custom focus area can be removed again', async () => {
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG,
        value: JSON.stringify({ instructions: '', categories: { 'pii leakage': true } }),
      },
    ];
    render();
    await openSettings();

    await userEvent.click(screen.getByRole('button', { name: 'Remove custom category pii leakage' }));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(written).toHaveLength(1));
    expect(savedSettings().categories['pii leakage']).toBeUndefined();
  });

  test('deselecting every focus area says what the run will do instead', async () => {
    render();
    await openSettings();

    for (const area of ['Correctness', 'Latency', 'Execution', 'Adherence', 'Relevance', 'Safety', 'Cost']) {
      await userEvent.click(screen.getByRole('button', { name: area }));
    }

    // Silently detecting everything would read as the settings being ignored.
    expect(screen.getByText('With no focus area selected, detection considers all of them.')).toBeInTheDocument();
  });

  test('a failed save keeps the modal open with the draft intact', async () => {
    saveFails = true;
    render();
    const priorities = await openSettings();

    await userEvent.type(priorities, 'Worth keeping.');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(await screen.findByText('Could not save issue detection settings. Try again.')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: /Detection priorities/ })).toHaveValue('Worth keeping.');
  });

  test('a failed settings read offers a retry instead of a dead control', async () => {
    loadFails = true;
    render();

    // The error alert lives inside the modal, so a disabled gear with no retry left the settings
    // unreachable until a remount even after the server recovered.
    const retry = await screen.findByRole('button', {
      name: /retry loading issue detection settings/i,
    });
    expect(retry).not.toBeDisabled();
    expect(screen.queryByRole('button', { name: 'Issue detection settings' })).not.toBeInTheDocument();

    loadFails = false;
    experimentTags = [
      {
        key: MLFLOW_ISSUE_DETECTION_PRIORITIES_TAG,
        value: JSON.stringify({ instructions: 'Only report high severity.', categories: {} }),
      },
    ];
    await userEvent.click(retry);

    // Editing is reachable again, and against what is really stored.
    await userEvent.click(await screen.findByRole('button', { name: 'Issue detection settings' }));
    expect(await screen.findByRole('textbox', { name: /Detection priorities/ })).toHaveValue(
      'Only report high severity.',
    );
  });

  test('a failed read cannot be saved over', async () => {
    loadFails = true;
    render();

    await screen.findByRole('button', { name: /retry loading issue detection settings/i });
    // No modal to open means no Save, so the defaults shown against a failed read cannot replace
    // whatever is stored.
    expect(screen.queryByRole('button', { name: 'Save' })).not.toBeInTheDocument();
    expect(written).toEqual([]);
  });
});
