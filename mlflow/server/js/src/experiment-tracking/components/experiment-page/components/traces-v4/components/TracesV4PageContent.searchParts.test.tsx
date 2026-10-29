import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { shouldEnableNewEvalRunsUi } from '@mlflow/mlflow/src/common/utils/FeatureUtils';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import {
  renderPage,
  findTraceRow,
  openDisplaySubmenu,
  selectSubmenuItem,
  state,
} from '../test-utils/tracesV4PageContentTestBed';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// Saved views read experiment tags through Apollo; stub it as the other page tests do.
jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));

const searchBox = () => screen.getByRole('combobox', { name: 'Type something to use full text search' });
const typeQuery = (value: string) => fireEvent.change(searchBox(), { target: { value } });
// rc-select reads the legacy keyCode for Enter.
const pressEnter = () => fireEvent.keyDown(searchBox(), { key: 'Enter', code: 'Enter', keyCode: 13, which: 13 });
const lastFilter = () => state.searchCalls[state.searchCalls.length - 1]?.filter ?? '';

describe('TracesV4PageContent search parts', () => {
  beforeEach(() => {
    state.pages = {
      '': {
        traces: [makeTrace('tr-000', { assessments: [makeFeedbackAssessment('relevance', 4)] })],
        next_page_token: undefined,
      },
    };
  });

  test('with the new eval runs UI on, sorts by state', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const user = userEvent.setup();
    renderPage();
    await findTraceRow('tr-000');

    await openDisplaySubmenu(user, /^Sort/);
    await selectSubmenuItem('menuitemradio', 'State (descending)');

    await waitFor(() => expect(state.searchCalls.some((call) => call.order_by?.[0] === 'status DESC')).toBe(true));
  }, 20000);

  test('with the new eval runs UI on, a numeric feedback filter compares as a number', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    renderPage();
    await findTraceRow('tr-000');

    typeQuery("feedback.relevance = '4'");
    pressEnter();

    await waitFor(() => expect(lastFilter()).toContain('feedback.`relevance` >= 4 AND feedback.`relevance` <= 4'));
  }, 20000);

  test('with the new eval runs UI on, a numeric-looking value on a text feedback stays literal', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    state.pages = {
      '': {
        traces: [makeTrace('tr-000', { assessments: [makeFeedbackAssessment('rating', '5')] })],
        next_page_token: undefined,
      },
    };
    renderPage();
    await findTraceRow('tr-000');

    typeQuery("feedback.rating = '5'");
    pressEnter();

    await waitFor(() => expect(lastFilter()).toContain("feedback.`rating` = '5'"));
    expect(lastFilter()).not.toContain('>=');
  }, 20000);

  test('with the new eval runs UI off, state is not offered as a sort', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    const user = userEvent.setup();
    renderPage();
    await findTraceRow('tr-000');

    await openDisplaySubmenu(user, /^Sort/);
    expect(await screen.findByRole('menuitemradio', { name: 'Duration (descending)' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitemradio', { name: 'State (descending)' })).not.toBeInTheDocument();
  }, 20000);
});
