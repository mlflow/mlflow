import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FilterOp, type TraceFilterModel } from '@databricks/web-shared/traces-table';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { renderPage, findTraceRow, state, env, URL } from '../test-utils/tracesV4PageContentTestBed';

jest.mock('@mlflow/mlflow/src/common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('@mlflow/mlflow/src/common/utils/FeatureUtils')>(
    '@mlflow/mlflow/src/common/utils/FeatureUtils',
  ),
  shouldEnableNewEvalRunsUi: () => true,
}));
jest.mock('@mlflow/mlflow/src/experiment-tracking/hooks/useExperimentQuery', () => ({
  useGetExperimentQuery: () => ({ data: { tags: [] }, refetch: () => Promise.resolve({}) }),
}));
jest.mock('./SqlFilterEditor', () => ({
  SqlFilterEditor: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <textarea aria-label="Trace filter predicate" value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}));
const basic = { field: 'duration', operator: FilterOp.GREATER_THAN, value: '1000' };
const sql = { field: 'sql_where', operator: FilterOp.EQUALS, value: "tags.env LIKE 'qa%'" };
const editor = () => screen.getByRole('textbox', { name: 'Trace filter predicate' });
const initialUrl = (filters: TraceFilterModel) =>
  `${URL}?tag=team%3Dml&filters=${encodeURIComponent(JSON.stringify(filters))}`;
const filters = () => JSON.parse(new URLSearchParams(env.lastSearch).get('filters') ?? '[]');

describe('trace SQL state and search', () => {
  beforeEach(() => {
    state.pages = { '': { traces: [makeTrace('tr-000')] } };
  });

  test('editing a SQL chip preserves Basic siblings and URL tags, including AND additions', async () => {
    const user = userEvent.setup();
    renderPage({ initialUrl: initialUrl([basic, sql]) });
    await findTraceRow('tr-000');
    await user.click(screen.getByRole('button', { name: `Edit filter SQL: ${sql.value}` }));
    expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue(sql.value);
    const next = "tags.env LIKE 'qa%' AND tags.release IS NOT NULL";
    fireEvent.change(editor(), { target: { value: next } });
    await user.click(screen.getByRole('button', { name: 'Apply filter' }));
    await waitFor(() => expect(filters()).toEqual([basic, { ...sql, value: next }]));
    expect(new URLSearchParams(env.lastSearch).getAll('tag')).toEqual(['team=ml']);
    await waitFor(() => expect(state.searchCalls.at(-1)?.filter).toContain(next));
    expect(state.searchCalls.at(-1)?.filter).toContain('attributes.execution_time_ms > 1000');
    expect(state.searchCalls.at(-1)?.filter).toContain("tags.team = 'ml'");
    await user.click(screen.getByRole('button', { name: `Remove filter SQL: ${next}` }));
    await waitFor(() => expect(filters()).toEqual([basic]));
    expect(new URLSearchParams(env.lastSearch).get('tag')).toBe('team=ml');
  }, 40000);

  test('converting one Basic chip to SQL preserves sibling predicates and tags', async () => {
    const user = userEvent.setup();
    renderPage({ initialUrl: initialUrl([basic, sql]) });
    await findTraceRow('tr-000');
    await user.click(screen.getByRole('button', { name: /Edit filter latency/ }));
    await user.click(await screen.findByRole('tab', { name: 'SQL' }));
    expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue(
      'attributes.execution_time_ms > 1000',
    );
    fireEvent.change(editor(), { target: { value: 'attributes.execution_time_ms > 2000' } });
    await user.click(screen.getByRole('button', { name: 'Apply filter' }));
    await waitFor(() => expect(filters()).toEqual([{ ...basic, value: '2000' }, sql]));
    expect(new URLSearchParams(env.lastSearch).get('tag')).toBe('team=ml');
  }, 30000);

  test('a cancelled SQL edit leaves the URL unchanged', async () => {
    const user = userEvent.setup();
    renderPage({ initialUrl: initialUrl([basic, sql]) });
    await findTraceRow('tr-000');
    const before = env.lastSearch;
    await user.click(screen.getByRole('button', { name: `Edit filter SQL: ${sql.value}` }));
    fireEvent.change(await screen.findByRole('textbox', { name: 'Trace filter predicate' }), {
      target: { value: 'attributes.status = bogus' },
    });
    await user.keyboard('{Escape}');
    expect(env.lastSearch).toBe(before);
    await user.click(screen.getByRole('button', { name: `Edit filter SQL: ${sql.value}` }));
    expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue(sql.value);
  }, 30000);
  test.each([FilterOp.EQUALS, FilterOp.NOT_EQUALS])(
    'does not invent numeric assessment conditions for an incomplete %s draft',
    async (operator) => {
      state.pages = { '': { traces: [makeTrace('tr-000', { assessments: [makeFeedbackAssessment('score', 4)] })] } };
      renderPage({ initialUrl: initialUrl([{ field: 'assessment', key: 'score', operator, value: '4' }]) });
      await findTraceRow('tr-000');
      await waitFor(() =>
        expect(state.searchCalls.at(-1)?.filter).toContain(
          operator === FilterOp.EQUALS
            ? 'feedback.`score` >= 4 AND feedback.`score` <= 4'
            : "feedback.`score` != '4.0'",
        ),
      );
      await userEvent.click(screen.getByRole('button', { name: `Edit filter feedback.score ${operator} '4'` }));
      await userEvent.click(await screen.findByRole('combobox', { name: 'Filter field' }));
      await userEvent.click(await screen.findByRole('option', { name: 'Tag' }));
      await userEvent.click(screen.getByRole('combobox', { name: 'Filter field' }));
      await userEvent.click(await screen.findByRole('option', { name: 'Feedback' }));
      await userEvent.click(screen.getByRole('combobox', { name: 'Filter key' }));
      await userEvent.click(await screen.findByRole('option', { name: 'score' }));
      if (operator === FilterOp.NOT_EQUALS) {
        await userEvent.click(screen.getByRole('combobox', { name: 'Filter operator' }));
        await userEvent.click(await screen.findByRole('option', { name: '!=' }));
      }
      await userEvent.click(screen.getByRole('tab', { name: 'SQL' }));
      expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue('');
    },
    30000,
  );
  test('preserves OR alternatives when converting a Basic sibling to SQL', async () => {
    const oldOr = "tags.env = 'qa' OR tags.env = 'prod'";
    const nextOr = "attributes.status = 'ERROR' OR tags.release = 'canary'";
    renderPage({ initialUrl: initialUrl([basic, { ...sql, value: oldOr }]) });
    await findTraceRow('tr-000');
    await userEvent.click(screen.getByRole('button', { name: /Edit filter latency/ }));
    await userEvent.click(await screen.findByRole('tab', { name: 'SQL' }));
    fireEvent.change(editor(), { target: { value: nextOr } });
    await userEvent.click(screen.getByRole('button', { name: 'Apply filter' }));
    await waitFor(() => expect(filters()).toEqual([{ ...sql, value: `(${oldOr}) AND (${nextOr})` }]));
    expect(new URLSearchParams(env.lastSearch).get('tag')).toBe('team=ml');
  }, 30000);

  test('groups an OR chip with its siblings in the full SQL editor', async () => {
    const expression = "tags.env = 'qa' OR tags.env = 'prod'";
    renderPage({ initialUrl: initialUrl([basic, { ...sql, value: expression }]) });
    await findTraceRow('tr-000');
    await userEvent.click(screen.getByRole('combobox', { name: 'Type something to use full text search' }));
    await userEvent.click(await screen.findByRole('button', { name: 'Add a Filter' }));
    await userEvent.click(await screen.findByRole('tab', { name: 'SQL' }));
    expect(await screen.findByRole('textbox', { name: 'Trace filter predicate' })).toHaveValue(
      `attributes.execution_time_ms > 1000 AND (${expression}) AND tags.team = 'ml'`,
    );
  }, 30000);
});
