import { describe, expect, jest, test } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { ExpectationAssessment, ModelTraceInfoV3 } from '@databricks/web-shared/model-trace-explorer';
import {
  FilterOp,
  TraceFilterButton,
  type FilterClause,
  type FilterFieldDef,
  type TraceFilterModel,
} from '@databricks/web-shared/traces-table';
import { slowlyTypeEachKey } from '@databricks/web-shared/test-utils/slowlyTypeEachKey';
import { QueryClient, QueryClientProvider } from '@databricks/web-shared/query-client';
import { IntlProvider } from 'react-intl';
import { useState } from 'react';
import userEvent from '@testing-library/user-event';
import { makeFeedbackAssessment, makeTrace } from '../test-utils/mockTraces';
import { compileFilterModel } from '../utils/filterModel';
import { TracesV4Search, type TracesV4SearchProps } from './TracesV4Search';

const makeExpectation = (name: string, value: string | number | boolean): ExpectationAssessment => ({
  assessment_id: `expectation-${name}-${value}`,
  assessment_name: name,
  trace_id: 'trace-1',
  source: { source_type: 'HUMAN', source_id: 'test-user' },
  create_time: '2025-01-01T00:00:00Z',
  last_update_time: '2025-01-01T00:00:00Z',
  expectation: { value },
});

const observedTrace = makeTrace('trace-1', {
  execution_duration: '1.2s',
  state: 'OK',
  tags: { 'mlflow.traceName': 'chat', env: 'prod' },
  trace_metadata: {
    'mlflow.trace.session': 'session-1',
    'mlflow.source.name': 'app.py',
    'mlflow.trace.user': 'alice',
  },
  assessments: [makeFeedbackAssessment('quality', 'pass')],
});

interface SearchHarnessProps extends Pick<
  TracesV4SearchProps,
  'filterModel' | 'onRemoveFilter' | 'onOpenFilter' | 'onRunTextSearch' | 'onMinimumWidthChange'
> {
  experimentId?: string;
  traces?: ModelTraceInfoV3[];
  onChange?: (value: string) => void;
  onClear?: () => void;
  onSubmit?: () => void;
  onApplyFilter?: (clause: FilterClause) => void;
}

const SearchHarness = ({
  traces = [observedTrace],
  onChange = () => {},
  onClear = () => {},
  onSubmit = () => {},
  onApplyFilter = () => {},
  experimentId = 'experiment-1',
  ...searchProps
}: SearchHarnessProps) => {
  const [value, setValue] = useState('');
  return (
    <TracesV4Search
      {...searchProps}
      key={experimentId}
      traces={traces}
      value={value}
      onChange={(nextValue) => {
        setValue(nextValue);
        onChange(nextValue);
      }}
      onClear={() => {
        setValue('');
        onClear();
      }}
      onSubmit={onSubmit}
      onApplyFilter={onApplyFilter}
    />
  );
};

const renderSearch = (props: SearchHarnessProps = {}) =>
  render(<SearchHarness {...props} />, {
    wrapper: ({ children }) => (
      <IntlProvider locale="en">
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
          <DesignSystemProvider>{children}</DesignSystemProvider>
        </QueryClientProvider>
      </IntlProvider>
    ),
  });

const searchBox = () => screen.getByRole('combobox');
// Du Bois names the input's clear icon `close-circle`.
const clearButton = () => screen.getByRole('button', { name: 'close-circle' });
// This rc-select version exposes only its active neighborhood as ARIA options.
// eslint-disable-next-line testing-library/no-node-access -- rc-select only exposes the active neighborhood through ARIA.
const visibleOptions = () => Array.from(document.querySelectorAll('[class*="-select-item-option"][aria-label]'));
// rc-select reads the legacy keyCode/which properties for option navigation.
const pressKey = (key: 'ArrowDown' | 'Enter') => {
  const keyCode = key === 'ArrowDown' ? 40 : 13;
  fireEvent.keyDown(searchBox(), { key, code: key, keyCode, which: keyCode });
};
const queryText = (text: string) => (_: string, element: Element | null) =>
  element?.tagName === 'CODE' && element.textContent === text;

const selectLatencyGreaterThanOrEqual = async () => {
  const user = userEvent.setup();
  await slowlyTypeEachKey(searchBox(), 'latency');
  await user.click(await screen.findByText(queryText('latency >=')));
  return user;
};

describe('TracesV4Search', () => {
  test('clearing search text leaves filter pills alone', async () => {
    const user = userEvent.setup();
    renderSearch({ filterModel: [{ field: 'state', operator: FilterOp.EQUALS, value: 'OK' }] });
    await slowlyTypeEachKey(searchBox(), 'unmatched');
    await user.click(clearButton());
    expect(searchBox()).toHaveValue('');
    expect(screen.getByRole('button', { name: "Edit filter state = 'OK'" })).toBeInTheDocument();
  });

  test('shows Add a filter at rest and text search guidance on focus', async () => {
    const user = userEvent.setup();
    renderSearch();
    expect(searchBox()).toHaveAttribute('placeholder', 'Add a filter');
    await user.click(searchBox());
    expect(searchBox()).toHaveAttribute('placeholder', 'Type something to use full text search');
    await user.tab({ shift: true });
    expect(searchBox()).toHaveAttribute('placeholder', 'Add a filter');
  });

  test('shows at most five observed examples and applies a clicked example immediately once', async () => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onApplyFilter, onSubmit });

    await user.click(searchBox());

    expect(screen.getByText(queryText('latency >= 1200 ms'))).toBeInTheDocument();
    expect(screen.getByText(queryText("feedback.quality = 'pass'"))).toBeInTheDocument();
    expect(screen.getByText(queryText("tag.env = 'prod'"))).toBeInTheDocument();
    expect(screen.getByText(queryText("state = 'OK'"))).toBeInTheDocument();
    expect(screen.getByText(queryText("trace_name = 'chat'"))).toBeInTheDocument();
    expect(screen.queryByText(queryText("session = 'session-1'"))).not.toBeInTheDocument();

    await user.click(screen.getByText(queryText('latency >= 1200 ms')));
    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');
    expect(onApplyFilter).toHaveBeenCalledTimes(1);
    expect(onSubmit).not.toHaveBeenCalled();
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '1200',
    });
  });

  test('keeps latency field, operator, and seconds value editable until Enter', async () => {
    const onApplyFilter = jest.fn();
    renderSearch({ traces: [], onApplyFilter });

    const user = await selectLatencyGreaterThanOrEqual();
    expect(searchBox()).toHaveValue('latency >= ');
    expect(searchBox()).toHaveAttribute('placeholder', 'Type something to use full text search');
    expect(screen.getByRole('button', { name: 'Add a Filter' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Remove filter/ })).not.toBeInTheDocument();
    await slowlyTypeEachKey(searchBox(), '2 s');
    expect(searchBox()).toHaveValue('latency >= 2 s');
    await user.keyboard('{Backspace}{Backspace}{Backspace}3 s');
    expect(searchBox()).toHaveValue('latency >= 3 s');
    expect(onApplyFilter).not.toHaveBeenCalled();
    await user.keyboard('{Enter}');
    expect(onApplyFilter).toHaveBeenCalledTimes(1);
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '3000',
    });
  });

  test('uses the plain text fallback when Enter is pressed for an unmatched query', async () => {
    const onSubmit = jest.fn();
    renderSearch({ onSubmit });

    await slowlyTypeEachKey(searchBox(), 'unmatched text');
    expect(screen.getByText('Search text “unmatched text”')).toBeInTheDocument();

    fireEvent.keyDown(searchBox(), {
      key: 'Enter',
      code: 'Enter',
      keyCode: 13,
      which: 13,
    });

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  test.each([
    ['latency', 'latency >='],
    ['state', 'state ='],
    ['feedback', 'feedback.quality'],
  ])('submits exact field word %s as literal text on Enter while offering completions', async (query, completion) => {
    const onSubmit = jest.fn();
    const onApplyFilter = jest.fn();
    renderSearch({ onSubmit, onApplyFilter });

    await slowlyTypeEachKey(searchBox(), query);
    expect(screen.getByRole('button', { name: 'Search' })).toBeEnabled();
    expect(screen.getByText(`Search text “${query}”`)).toBeInTheDocument();
    expect(screen.getByText(queryText(completion))).toBeInTheDocument();
    pressKey('Enter');

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onApplyFilter).not.toHaveBeenCalled();
  });

  test('Search button submits a bare field as text, while a complete clause still applies as a filter', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    const onApplyFilter = jest.fn();
    renderSearch({ onSubmit, onApplyFilter });

    await slowlyTypeEachKey(searchBox(), 'latency');
    await user.click(screen.getByRole('button', { name: 'Search' }));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onApplyFilter).not.toHaveBeenCalled();

    await user.clear(searchBox());
    await slowlyTypeEachKey(searchBox(), 'latency >= 2 s');
    await user.click(screen.getByRole('button', { name: 'Search' }));
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '2000',
    });
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  test('keeps the literal option visible within five rows when recent searches match a bare field', () => {
    renderSearch();
    for (const query of ['first latency note', 'second latency note']) {
      fireEvent.change(searchBox(), { target: { value: query } });
      pressKey('Enter');
      fireEvent.change(searchBox(), { target: { value: '' } });
    }

    fireEvent.change(searchBox(), { target: { value: 'latency' } });
    expect(searchBox()).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByText('first latency note')).toBeInTheDocument();
    expect(screen.getByText('second latency note')).toBeInTheDocument();
    expect(screen.getByText(queryText('latency >'))).toBeInTheDocument();
    expect(screen.getByText('Search text “latency”')).toBeInTheDocument();
    expect(visibleOptions()).toHaveLength(5);
  });

  test('applies a complete dotted feedback query through the filter callback', async () => {
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onApplyFilter, onSubmit });

    await slowlyTypeEachKey(searchBox(), 'feedback.groundedness = pass');
    fireEvent.keyDown(searchBox(), {
      key: 'Enter',
      code: 'Enter',
      keyCode: 13,
      which: 13,
    });

    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'assessment',
      key: 'groundedness',
      operator: FilterOp.EQUALS,
      value: 'pass',
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test('applies whitespace-tokenized feedback queries and keeps incomplete syntax local', async () => {
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    const rendered = renderSearch({ onApplyFilter, onSubmit });

    await slowlyTypeEachKey(searchBox(), 'feedback groundedness = pass');
    fireEvent.keyDown(searchBox(), {
      key: 'Enter',
      code: 'Enter',
      keyCode: 13,
      which: 13,
    });
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'assessment',
      key: 'groundedness',
      operator: FilterOp.EQUALS,
      value: 'pass',
    });

    rendered.unmount();
    renderSearch({ onApplyFilter, onSubmit });
    await slowlyTypeEachKey(searchBox(), 'feedback groundedness =');
    fireEvent.keyDown(searchBox(), {
      key: 'Enter',
      code: 'Enter',
      keyCode: 13,
      which: 13,
    });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test.each<{
    query: string;
    optionName: string;
    expectedClause: FilterClause;
  }>([
    {
      query: 'prod',
      optionName: "tag.env = 'prod'",
      expectedClause: {
        field: 'tag',
        key: 'env',
        operator: FilterOp.EQUALS,
        value: 'prod',
      },
    },
    {
      query: 'pass',
      optionName: "feedback.quality = 'pass'",
      expectedClause: {
        field: 'assessment',
        key: 'quality',
        operator: FilterOp.EQUALS,
        value: 'pass',
      },
    },
  ])('applies observed $query suggestion once on click', async ({ query, optionName, expectedClause }) => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onApplyFilter, onSubmit });

    await slowlyTypeEachKey(searchBox(), query);
    await user.click(await screen.findByText(queryText(optionName)));

    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');
    expect(onApplyFilter).toHaveBeenCalledTimes(1);
    expect(onSubmit).not.toHaveBeenCalled();
    expect(onApplyFilter).toHaveBeenCalledWith(expectedClause);
  });

  test('clears the controlled query and closes suggestions with Escape', async () => {
    const user = userEvent.setup();
    const onClear = jest.fn();
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onClear, onApplyFilter, onSubmit });

    await slowlyTypeEachKey(searchBox(), 'unmatched');
    expect(searchBox()).toHaveAttribute('aria-expanded', 'true');

    await user.keyboard('{Escape}');
    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');

    await user.click(clearButton());
    expect(onClear).toHaveBeenCalledTimes(1);
    expect(searchBox()).toHaveValue('');
    expect(onApplyFilter).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test.each(['not-a-number', '1.5 ms', '0.0001 s'])('does not offer invalid latency %s', async (value) => {
    const onApplyFilter = jest.fn();
    renderSearch({ traces: [], onApplyFilter });

    await selectLatencyGreaterThanOrEqual();
    await slowlyTypeEachKey(searchBox(), value);

    expect(searchBox()).toHaveValue(`latency >= ${value}`);
    expect(searchBox()).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('button', { name: 'Add a Filter' })).toBeInTheDocument();
    expect(screen.getByText(`Search text “latency >= ${value}”`)).toBeInTheDocument();
    fireEvent.keyDown(searchBox(), {
      key: 'Enter',
      code: 'Enter',
      keyCode: 13,
      which: 13,
    });
    expect(onApplyFilter).not.toHaveBeenCalled();
  });
});

describe('feedback and expectation search UI', () => {
  test.each([
    ['feedback', 'assessment', 'pass', 'gold'],
    ['expectation', 'expectation', 'gold', 'pass'],
  ])('applies the distinct %s value for a shared name', async (namespace, field, value, otherValue) => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({
      onApplyFilter,
      onSubmit,
      traces: [
        makeTrace('shared', {
          assessments: [makeFeedbackAssessment('quality', 'pass'), makeExpectation('quality', 'gold')],
        }),
      ],
    });
    await slowlyTypeEachKey(searchBox(), `${namespace}.quality =`);
    expect(screen.queryByText(queryText(`${namespace}.quality = '${otherValue}'`))).not.toBeInTheDocument();
    expect(screen.queryByText(queryText("assessment.quality = 'pass'"))).not.toBeInTheDocument();
    await user.click(await screen.findByText(queryText(`${namespace}.quality = '${value}'`)));
    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');
    expect(onApplyFilter).toHaveBeenCalledTimes(1);
    expect(onSubmit).not.toHaveBeenCalled();
    expect(onApplyFilter).toHaveBeenCalledWith({
      field,
      key: 'quality',
      operator: FilterOp.EQUALS,
      value,
    });
    expect(compileFilterModel([onApplyFilter.mock.calls[0][0] as FilterClause])).toEqual([
      `${namespace}.\`quality\` = '${value}'`,
    ]);
  });

  test('does not offer or apply assessment syntax', async () => {
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onApplyFilter, onSubmit });
    await slowlyTypeEachKey(searchBox(), 'assessment.quality = pass');
    expect(screen.queryByText(queryText("assessment.quality = 'pass'"))).not.toBeInTheDocument();
    fireEvent.keyDown(searchBox(), {
      key: 'Enter',
      code: 'Enter',
      keyCode: 13,
      which: 13,
    });
    expect(onApplyFilter).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

describe('editable query keyboard and operator selection', () => {
  test('ArrowDown Enter applies a complete suggested value immediately once', async () => {
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onApplyFilter, onSubmit });
    await slowlyTypeEachKey(searchBox(), 'feedback.quality = pa');
    pressKey('ArrowDown'); // The typed value is itself a complete query.
    pressKey('ArrowDown');
    pressKey('Enter');
    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');
    expect(onSubmit).not.toHaveBeenCalled();
    expect(onApplyFilter).toHaveBeenCalledTimes(1);
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'assessment',
      key: 'quality',
      operator: FilterOp.EQUALS,
      value: 'pass',
    });
  });

  test('keyboard field and operator selections only insert text', async () => {
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ traces: [], onApplyFilter, onSubmit });
    await slowlyTypeEachKey(searchBox(), 'laten');
    pressKey('ArrowDown');
    pressKey('Enter');
    expect(searchBox()).toHaveValue('latency ');
    expect(onApplyFilter).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
    pressKey('ArrowDown');
    pressKey('Enter');
    expect((searchBox() as HTMLInputElement).value).toMatch(/^latency [<>]=? $/);
    expect(onApplyFilter).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test('clicked field and operator suggestions only insert editable text', async () => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ traces: [], onApplyFilter, onSubmit });

    await slowlyTypeEachKey(searchBox(), 'laten');
    await user.click(await screen.findByText(queryText('latency…')));
    expect(searchBox()).toHaveValue('latency ');
    expect(onApplyFilter).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();

    await user.click(await screen.findByText(queryText('latency >=')));
    expect(searchBox()).toHaveValue('latency >= ');
    await slowlyTypeEachKey(searchBox(), '2 s');
    expect(searchBox()).toHaveValue('latency >= 2 s');
    expect(onApplyFilter).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  test('search button applies the edited complete query exactly once', async () => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    renderSearch({ onApplyFilter });
    await slowlyTypeEachKey(searchBox(), 'latency >= 2 s');
    await user.click(screen.getByRole('button', { name: 'Search' }));
    expect(onApplyFilter).toHaveBeenCalledTimes(1);
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '2000',
    });
  });

  test('typing feedback immediately offers keys', async () => {
    renderSearch();
    await slowlyTypeEachKey(searchBox(), 'feedback');
    expect(screen.getByText(queryText('feedback.quality'))).toBeInTheDocument();
  });

  test('offers and applies != for feedback and expectation, and no null checks', async () => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    const onSubmit = jest.fn();
    renderSearch({ onApplyFilter, onSubmit });
    for (const namespace of ['feedback', 'expectation']) {
      await user.clear(searchBox());
      await slowlyTypeEachKey(searchBox(), `${namespace}.quality`);
      expect(screen.getByText(queryText(`${namespace}.quality =`))).toBeInTheDocument();
      expect(screen.getByText(queryText(`${namespace}.quality !=`))).toBeInTheDocument();
      expect(screen.queryByText(queryText(`${namespace}.quality IS NULL`))).not.toBeInTheDocument();
      await slowlyTypeEachKey(searchBox(), ' != pass');
      await user.keyboard('{Enter}');
    }
    expect(onSubmit).not.toHaveBeenCalled();
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'assessment',
      key: 'quality',
      operator: FilterOp.NOT_EQUALS,
      value: 'pass',
    });
    expect(onApplyFilter).toHaveBeenCalledWith({
      field: 'expectation',
      key: 'quality',
      operator: FilterOp.NOT_EQUALS,
      value: 'pass',
    });
  });
});

describe('unified search shell', () => {
  test('the footer is reachable by keyboard and opens the editor after closing suggestions', async () => {
    const user = userEvent.setup();
    const onOpenFilter = jest.fn();
    renderSearch({ traces: [], onOpenFilter });
    await user.click(searchBox());
    await user.tab();
    expect(screen.getByRole('button', { name: 'Add a Filter' })).toHaveFocus();
    await user.keyboard('{Enter}');
    expect(onOpenFilter).toHaveBeenCalledTimes(1);
  });

  test('offers an explicit text search when no filter suggestions match', async () => {
    const user = userEvent.setup();
    const onOpenFilter = jest.fn((_index?: number) => expect(searchBox()).toHaveAttribute('aria-expanded', 'false'));
    const onSubmit = jest.fn();
    renderSearch({ traces: [], onOpenFilter, onSubmit });
    await user.click(searchBox());
    expect(screen.getByText('Suggestions')).toBeInTheDocument();
    expect(screen.queryByText('Recent')).not.toBeInTheDocument();
    expect(visibleOptions().length).toBeLessThanOrEqual(5);
    fireEvent.change(searchBox(), { target: { value: 'latency >= nope' } });
    expect(screen.getByText('Suggestions')).toBeInTheDocument();
    expect(screen.getByText('Search text “latency >= nope”')).toBeInTheDocument();
    expect(screen.queryByText(/Search with AI/)).not.toBeInTheDocument();
    await user.click(screen.getByText('Search text “latency >= nope”'));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    await user.tab();
    await user.click(searchBox());
    await user.click(screen.getByRole('button', { name: 'Add a Filter' }));
    expect(onOpenFilter).toHaveBeenCalledTimes(1);
    expect(onOpenFilter).toHaveBeenCalledWith(undefined);
  });

  test('applied pills use canonical labels and separate edit/remove buttons without opening suggestions', async () => {
    const user = userEvent.setup();
    const onOpenFilter = jest.fn();
    const onRemoveFilter = jest.fn();
    const longValue = 'a'.repeat(200);
    renderSearch({
      filterModel: [
        {
          field: 'duration',
          operator: FilterOp.GREATER_THAN_OR_EQUALS,
          value: '',
        },
        {
          field: 'assessment',
          key: 'quality',
          operator: FilterOp.EQUALS,
          value: longValue,
        },
        {
          field: 'duration',
          operator: FilterOp.GREATER_THAN_OR_EQUALS,
          value: '1200',
        },
      ],
      onOpenFilter,
      onRemoveFilter,
    });
    const label = `feedback.quality = '${longValue}'`;
    const edit = screen.getByRole('button', { name: `Edit filter ${label}` });
    const remove = screen.getByRole('button', {
      name: `Remove filter ${label}`,
    });
    expect(edit).toHaveAttribute('title', label);
    expect(edit).not.toContainElement(remove);
    await user.hover(edit);
    expect(await screen.findByRole('tooltip')).toHaveTextContent(label);
    await user.unhover(edit);
    expect(screen.getByRole('button', { name: 'Edit filter latency >= 1200 ms' })).toBeInTheDocument();
    expect(screen.queryByText(/assessment/)).not.toBeInTheDocument();
    await user.tab();
    expect(edit).toHaveFocus();
    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');
    await user.keyboard('{Enter}');
    expect(onOpenFilter).toHaveBeenCalledWith(1);
    await user.click(remove);
    expect(onRemoveFilter).toHaveBeenCalledWith(1);
    expect(onOpenFilter).toHaveBeenCalledTimes(1);
    expect(searchBox()).toHaveAttribute('aria-expanded', 'false');
  });
});

describe('memory-only recent searches', () => {
  const changeQuery = async (value: string) => {
    await userEvent.clear(searchBox());
    if (value) await slowlyTypeEachKey(searchBox(), value);
  };

  test('keeps only submitted requests, deduplicates examples, and reapplies a recent filter', async () => {
    const user = userEvent.setup();
    const onApplyFilter = jest.fn();
    renderSearch({ onApplyFilter });
    await user.click(searchBox());
    await changeQuery('latency >= ');
    pressKey('Enter');
    await changeQuery('');
    expect(screen.queryByText('Recent')).not.toBeInTheDocument();
    await changeQuery('duration >= 1.2 s');
    pressKey('Enter');
    await changeQuery('');
    expect(screen.getByText('Recent')).toBeInTheDocument();
    expect(visibleOptions()).toHaveLength(5);
    expect(screen.getAllByText(queryText('latency >= 1200 ms'))).toHaveLength(1);
    await user.click(screen.getByText(queryText('latency >= 1200 ms')));
    expect(onApplyFilter).toHaveBeenCalledTimes(2);
    expect(onApplyFilter).toHaveBeenLastCalledWith({
      field: 'duration',
      operator: FilterOp.GREATER_THAN_OR_EQUALS,
      value: '1200',
    });
  });

  test('caps MRU history at five, reserves two rows, and promotes a repeated submission', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    const onRunTextSearch = jest.fn();
    renderSearch({ onSubmit, onRunTextSearch });
    await user.click(searchBox());
    for (let index = 0; index < 6; index++) {
      await changeQuery(`needle ${index}`);
      pressKey('Enter');
    }
    expect(onSubmit).toHaveBeenCalledTimes(6);
    await changeQuery('');
    expect(visibleOptions()).toHaveLength(5);
    expect(
      visibleOptions()
        .slice(0, 2)
        .map((option) => option.getAttribute('aria-label')),
    ).toEqual(['needle 5', 'needle 4']);
    await changeQuery('needle 0');
    expect(screen.queryByText('Recent')).not.toBeInTheDocument();
    await changeQuery('needle 1');
    expect(screen.getByText('Recent')).toBeInTheDocument();
    await user.click(screen.getByText('needle 1'));
    expect(onRunTextSearch).toHaveBeenCalledWith('needle 1');
    expect(onSubmit).toHaveBeenCalledTimes(6);
    await changeQuery('');
    expect(
      visibleOptions()
        .slice(0, 2)
        .map((option) => option.getAttribute('aria-label')),
    ).toEqual(['needle 1', 'needle 5']);
    await changeQuery('needle 1');
    pressKey('Enter');
    await changeQuery('needle 1');
    expect(visibleOptions()).toHaveLength(1);
  });

  test('fallback inserts recent text without submitting a stale controlled value', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    const onChange = jest.fn();
    renderSearch({ onSubmit, onChange });
    await user.click(searchBox());
    await changeQuery('random words');
    await user.click(screen.getByText('Search text “random words”'));
    await changeQuery('');
    await user.click(screen.getByText('random words'));
    expect(onChange).toHaveBeenLastCalledWith('random words');
    expect(searchBox()).toHaveValue('random words');
    expect(onSubmit).toHaveBeenCalledTimes(1);
    pressKey('Enter');
    expect(onSubmit).toHaveBeenCalledTimes(2);
  });

  test('history survives rerenders but is cleared by the root experiment key', async () => {
    const user = userEvent.setup();
    const rendered = renderSearch();
    await user.click(searchBox());
    await changeQuery('private query');
    pressKey('Enter');
    rendered.rerender(<SearchHarness />);
    await changeQuery('');
    expect(screen.getByText('Recent')).toBeInTheDocument();
    rendered.rerender(<SearchHarness experimentId="experiment-2" />);
    await user.click(searchBox());
    expect(screen.queryByText('Recent')).not.toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'private query' })).not.toBeInTheDocument();
  });

  test('empty Enter submits after clearing without adding empty or unsubmitted text to history', async () => {
    const user = userEvent.setup();
    const onSubmit = jest.fn();
    renderSearch({ onSubmit });
    await changeQuery('committed query');
    pressKey('Enter');
    await user.click(clearButton());
    pressKey('Enter');
    await changeQuery('unsubmitted query');
    await changeQuery('');
    expect(screen.getByText('committed query')).toBeInTheDocument();
    expect(screen.queryByRole('option', { name: 'unsubmitted query' })).not.toBeInTheDocument();
    expect(onSubmit).toHaveBeenCalledTimes(2);
  });
});

const EDITOR_FIELDS: FilterFieldDef[] = [
  { id: 'state', label: 'State', operators: [FilterOp.EQUALS], valueInput: 'text' },
];

const SearchWithFilterEditor = ({ onFilterChange }: { onFilterChange: (model: TraceFilterModel) => void }) => {
  const [filterModel, setFilterModel] = useState<TraceFilterModel>([]);
  const [value, setValue] = useState('');
  const renderSearchControl = (openEditor: (filterIndex?: number) => void) => (
    <TracesV4Search
      traces={[]}
      value={value}
      onChange={setValue}
      onClear={() => setValue('')}
      onSubmit={() => {}}
      onApplyFilter={(clause) => setFilterModel((previous) => [...previous, clause])}
      filterModel={filterModel}
      onOpenFilter={openEditor}
      onRemoveFilter={(index) => setFilterModel((previous) => previous.filter((_, i) => i !== index))}
    />
  );

  return (
    <IntlProvider locale="en">
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <DesignSystemProvider>
          <TraceFilterButton
            fields={EDITOR_FIELDS}
            filterModel={filterModel}
            activeCount={filterModel.length}
            onChange={(next) => {
              setFilterModel(next);
              onFilterChange(next);
            }}
            onClearAll={() => setFilterModel([])}
            renderTrigger={renderSearchControl}
          />
          <button>Display</button>
        </DesignSystemProvider>
      </QueryClientProvider>
    </IntlProvider>
  );
};

describe('search and real filter editor portal integration', () => {
  test('preserves reverse Tab and normal toolbar order after Escape', async () => {
    const user = userEvent.setup();
    render(<SearchWithFilterEditor onFilterChange={() => {}} />);
    const input = screen.getByRole('combobox', { name: 'Type something to use full text search' });
    await user.click(input);
    await user.paste('random words');
    await user.tab({ shift: true });
    expect(screen.getByRole('button', { name: 'Search' })).toHaveFocus();
    await user.click(input);
    await user.keyboard('{Escape}');
    expect(input).toHaveAttribute('aria-expanded', 'false');
    await user.tab();
    expect(screen.getByRole('button', { name: 'Display' })).toHaveFocus();
  });

  test.each(['click', 'keyboard'] as const)(
    '%s footer handoff applies a filter and edits the same pill',
    async (interaction) => {
      const user = userEvent.setup();
      const onFilterChange = jest.fn();
      render(<SearchWithFilterEditor onFilterChange={onFilterChange} />);
      const input = screen.getByRole('combobox', { name: 'Type something to use full text search' });
      await user.click(input);
      expect(input).toHaveAttribute('aria-expanded', 'true');
      expect(screen.getByText('Suggestions')).toBeInTheDocument();
      const footer = screen.getByRole('button', { name: 'Add a Filter' });
      if (interaction === 'keyboard') {
        await user.tab();
        expect(footer).toHaveFocus();
        await user.keyboard('{Enter}');
      } else {
        await user.click(footer);
      }

      const editorValue = await screen.findByRole('textbox', { name: 'Filter value' });
      await waitFor(() => expect(screen.getByRole('combobox', { name: 'Filter field' })).toHaveFocus());
      expect(input).toHaveAttribute('aria-expanded', 'false');
      await waitFor(() => expect(screen.queryByText(queryText('state…'))).not.toBeInTheDocument());
      await user.click(editorValue);
      await user.paste('OK');
      await user.click(screen.getByRole('button', { name: 'Apply filters' }));
      expect(onFilterChange).toHaveBeenCalledTimes(1);
      expect(onFilterChange).toHaveBeenLastCalledWith([{ field: 'state', operator: FilterOp.EQUALS, value: 'OK' }]);
      await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Filter value' })).not.toBeInTheDocument());
      const pill = screen.getByRole('button', { name: "Edit filter state = 'OK'" });
      expect(pill).toBeVisible();
      expect(input).toHaveAttribute('aria-expanded', 'false');

      await user.click(pill);
      const editValue = await screen.findByRole('textbox', { name: 'Filter value' });
      expect(editValue).toHaveValue('OK');
      expect(screen.getAllByRole('textbox', { name: 'Filter value' })).toHaveLength(1);
      expect(input).toHaveAttribute('aria-expanded', 'false');
      await user.clear(editValue);
      await user.paste('ERROR');
      await user.click(screen.getByRole('button', { name: 'Apply filter' }));
      expect(onFilterChange).toHaveBeenCalledTimes(2);
      expect(onFilterChange).toHaveBeenLastCalledWith([{ field: 'state', operator: FilterOp.EQUALS, value: 'ERROR' }]);
      expect(await screen.findByRole('button', { name: "Edit filter state = 'ERROR'" })).toBeVisible();
      expect(screen.queryByRole('button', { name: "Edit filter state = 'OK'" })).not.toBeInTheDocument();
      await waitFor(() => expect(screen.queryByRole('textbox', { name: 'Filter value' })).not.toBeInTheDocument());
    },
  );
});

describe('compact search layout', () => {
  test('keeps the search action before every applied pill and the editable input', () => {
    renderSearch({ filterModel: [{ field: 'state', operator: FilterOp.EQUALS, value: 'OK' }] });
    const search = screen.getByRole('button', { name: 'Search' });
    const edit = screen.getByRole('button', { name: "Edit filter state = 'OK'" });
    expect(search.compareDocumentPosition(edit) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(edit.compareDocumentPosition(searchBox()) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(searchBox()).toHaveAttribute('placeholder', 'Add a filter');
  });

  test('reports capped intrinsic pill widths, ignores available space, and cleans up measurement observers', () => {
    const onMinimumWidthChange = jest.fn();
    let availableWidth = 900;
    let firstPillWidth = 100;
    const observerRecords: { notify: () => void; disconnect: ReturnType<typeof jest.fn> }[] = [];
    const observerMock = jest.spyOn(window, 'ResizeObserver').mockImplementation((callback) => {
      const observer = { observe: jest.fn(), unobserve: jest.fn(), disconnect: jest.fn() };
      observerRecords.push({ notify: () => callback([], observer), disconnect: observer.disconnect });
      return observer;
    });
    const rectMock = jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (
      this: HTMLElement,
    ) {
      // JSDOM has no layout; model intrinsic pill sizes separately from available container space.
      // eslint-disable-next-line testing-library/no-node-access
      const label = this.tagName === 'SPAN' ? this.querySelector('button')?.getAttribute('aria-label') : undefined;
      const width =
        label === "Edit filter state = 'OK'" ? firstPillWidth : label?.startsWith('Edit filter') ? 250 : availableWidth;
      return { width, height: 24, top: 0, left: 0, bottom: 24, right: width, x: 0, y: 0, toJSON: () => ({}) };
    });
    const filterModel: TraceFilterModel = [
      { field: 'state', operator: FilterOp.EQUALS, value: 'OK' },
      { field: 'tag', key: 'env', operator: FilterOp.EQUALS, value: 'long value' },
    ];
    const notify = () => act(() => observerRecords.forEach((observer) => observer.notify()));
    let rendered: ReturnType<typeof renderSearch> | undefined;
    try {
      rendered = renderSearch({ filterModel, onMinimumWidthChange });
      // 24 icon + (100 + capped 200) pills + 3*4 gaps + 300 input + 8 padding + 2 border.
      expect(onMinimumWidthChange).toHaveBeenLastCalledWith(646);
      const calls = onMinimumWidthChange.mock.calls.length;
      availableWidth = 350;
      notify();
      fireEvent(window, new Event('resize'));
      expect(onMinimumWidthChange).toHaveBeenCalledTimes(calls);
      availableWidth = 1200;
      notify();
      expect(onMinimumWidthChange).toHaveBeenCalledTimes(calls);

      // Font/layout changes resize the non-shrinking pill itself.
      firstPillWidth = 120;
      notify();
      expect(onMinimumWidthChange).toHaveBeenLastCalledWith(666);
      notify();
      expect(onMinimumWidthChange).toHaveBeenCalledTimes(calls + 1);
      const previousObservers = [...observerRecords];
      rendered.rerender(
        <SearchHarness filterModel={filterModel.slice(0, 1)} onMinimumWidthChange={onMinimumWidthChange} />,
      );
      expect(onMinimumWidthChange).toHaveBeenLastCalledWith(462);
      expect(previousObservers.some((observer) => observer.disconnect.mock.calls.length > 0)).toBe(true);
      rendered.rerender(<SearchHarness filterModel={[]} onMinimumWidthChange={onMinimumWidthChange} />);
      expect(onMinimumWidthChange).toHaveBeenLastCalledWith(338);
      rendered.unmount();
      const callsAfterUnmount = onMinimumWidthChange.mock.calls.length;
      notify();
      fireEvent(window, new Event('resize'));
      expect(onMinimumWidthChange).toHaveBeenCalledTimes(callsAfterUnmount);
      expect(observerRecords.every((observer) => observer.disconnect.mock.calls.length > 0)).toBe(true);
    } finally {
      rendered?.unmount();
      rectMock.mockRestore();
      observerMock.mockRestore();
    }
  });
});
