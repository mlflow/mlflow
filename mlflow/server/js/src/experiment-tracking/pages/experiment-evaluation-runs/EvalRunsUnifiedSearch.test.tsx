import { useState } from 'react';

import { describe, expect, test } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen } from '../../../common/utils/TestUtils.react18';
import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import type { RunEntity } from '../../types';
import { EvalRunsFilterButton } from './EvalRunsFilterButton';
import { EvalRunsUnifiedSearch } from './EvalRunsUnifiedSearch';
import type { EvalRunsFilterClause } from './evalRunsFilterFields';

const run = {
  info: { runName: 'Example run', status: 'FINISHED' },
  data: { metrics: [], params: [], tags: [] },
} as unknown as RunEntity;

const runsData = {
  runInfos: [],
  runUuidsMatchingFilter: [],
  metricKeyList: [],
  paramKeyList: [],
  metricsList: [],
  paramsList: [],
  tagsList: [],
  modelVersionsByRunUuid: {},
  experimentTags: {},
  datasetsList: [],
} as ExperimentRunsSelectorResult;

const Harness = ({ runs = [run] }: { runs?: RunEntity[] }) => {
  const [searchFilter, setSearchFilter] = useState('');
  const [clauses, setClauses] = useState<EvalRunsFilterClause[]>([]);
  return (
    <>
      <EvalRunsFilterButton
        clauses={clauses}
        setClauses={setClauses}
        runsData={runsData}
        renderTrigger={(openFilter) => (
          <EvalRunsUnifiedSearch
            runs={runs}
            runsData={runsData}
            searchFilter={searchFilter}
            onSearchFilterChange={setSearchFilter}
            clauses={clauses}
            setClauses={setClauses}
            openFilter={openFilter}
          />
        )}
      />
      <output data-testid="compiled-search">{searchFilter}</output>
    </>
  );
};

describe('EvalRunsUnifiedSearch', () => {
  const codeText = (text: string) => (_content: string, element: Element | null) =>
    element?.tagName === 'CODE' && element.textContent === text;

  test('suggests run name, status, one metric, start time, and dataset from observed runs', async () => {
    const user = userEvent.setup();
    const observedRun = {
      ...run,
      info: { ...run.info, startTime: 123 },
      data: {
        metrics: [
          { key: 'correctness/mean', value: 1 },
          { key: 'agent_tokens/mean', value: 42 },
        ],
      },
      inputs: { datasetInputs: [{ dataset: { name: 'returns-dataset' } }] },
    } as RunEntity;
    renderWithDesignSystem(<Harness runs={[observedRun]} />);

    const search = screen.getByRole('combobox', { name: 'Search evaluation runs' });
    await user.click(search);
    for (const suggestion of [
      "attributes.run_name = 'Example run'",
      "attributes.status = 'FINISHED'",
      'metrics.`correctness/mean` = 1',
      'attributes.start_time = 123',
      "dataset.name = 'returns-dataset'",
    ]) {
      expect(screen.getByText(codeText(suggestion))).toBeInTheDocument();
    }
    expect(screen.queryByText(codeText('metrics.`agent_tokens/mean` = 42'))).not.toBeInTheDocument();

    await user.click(screen.getByText(codeText('attributes.start_time = 123')));
    expect(screen.getByRole('button', { name: 'Edit filter attributes.start_time = 123' })).toBeInTheDocument();
    await user.click(search);
    await user.click(screen.getByText(codeText("dataset.name = 'returns-dataset'")));
    expect(screen.getByRole('button', { name: "Edit filter dataset.name = 'returns-dataset'" })).toBeInTheDocument();
  });

  test('applies an observed filter selected from Suggestions', async () => {
    const user = userEvent.setup();
    renderWithDesignSystem(<Harness />);

    const search = screen.getByRole('combobox', { name: 'Search evaluation runs' });
    await user.click(search);
    await user.click(screen.getByText(codeText("attributes.run_name = 'Example run'")));

    expect(
      screen.getByRole('button', {
        name: "Edit filter attributes.run_name = 'Example run'",
      }),
    ).toBeInTheDocument();
    expect(search).toHaveValue('');
  });

  test('moves focus to Add a Filter before opening the editor', async () => {
    const user = userEvent.setup();
    renderWithDesignSystem(<Harness />);

    const search = screen.getByRole('combobox', { name: 'Search evaluation runs' });
    await user.click(search);
    await user.tab();
    expect(screen.getByRole('button', { name: 'Add a Filter' })).toHaveFocus();
    await user.keyboard('{Enter}');

    expect(search).toHaveAttribute('aria-expanded', 'false');
    expect(await screen.findByRole('combobox', { name: 'Filter field' })).toHaveFocus();
  });

  test('submits a valid raw SearchRuns expression for a known field', async () => {
    const user = userEvent.setup();
    renderWithDesignSystem(<Harness />);

    const search = screen.getByRole('combobox', { name: 'Search evaluation runs' });
    await user.click(search);
    await user.paste("attributes.run_name ILIKE 'foo'");
    await user.keyboard('{Enter}');

    expect(screen.getByTestId('compiled-search')).toHaveTextContent("attributes.run_name ILIKE 'foo'");
  });

  test.each([
    "metric.f1_score > 0.9 AND params.model = 'tree'",
    "dataset.name IN ('dataset1', 'dataset2')",
    "attributes.run_id IN ('a1b2c3d4', 'e5f6g7h8')",
    "tags.model_class LIKE 'sklearn.linear_model%'",
  ])('submits the search help example %s as a raw SearchRuns expression', async (expression) => {
    const user = userEvent.setup();
    renderWithDesignSystem(<Harness />);

    const search = screen.getByRole('combobox', { name: 'Search evaluation runs' });
    await user.click(search);
    await user.paste(expression);
    await user.keyboard('{Enter}');

    // Exact match: a run-name search would wrap the expression in `attributes.run_name ILIKE '%…%'`.
    expect(screen.getByTestId('compiled-search').textContent).toBe(expression);
  });

  test('shows the search syntax help with examples and a docs link', async () => {
    const user = userEvent.setup();
    renderWithDesignSystem(<Harness />);

    await user.click(screen.getByRole('button', { name: 'Search syntax help' }));

    expect(await screen.findByText(/Type part of a run name to search by name/)).toBeInTheDocument();
    expect(screen.getByText(/metrics\.rmse >= 0\.8/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Learn more' })).toHaveAttribute(
      'href',
      expect.stringContaining('search-runs'),
    );
  });

  test('owns suggestions, the filter editor, pills, run-name search, and recents in one control', async () => {
    const user = userEvent.setup();
    renderWithDesignSystem(<Harness />);

    const search = screen.getByRole('combobox', { name: 'Search evaluation runs' });
    expect(search).toHaveAttribute('placeholder', 'Add a filter');
    expect(screen.queryByRole('button', { name: 'Filters' })).not.toBeInTheDocument();

    await user.click(search);
    expect(screen.getByText('Suggestions')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Add a Filter' }));
    expect(search).toHaveAttribute('aria-expanded', 'false');
    const filterField = await screen.findByRole('combobox', { name: 'Filter field' });
    await user.click(filterField);
    await user.click(screen.getByRole('option', { name: 'Run name' }));

    await user.click(screen.getByRole('textbox', { name: 'Filter value' }));
    await user.paste('keep');
    await user.click(screen.getByRole('button', { name: 'Apply' }));
    expect(screen.getByRole('button', { name: "Edit filter attributes.run_name = 'keep'" })).toBeInTheDocument();

    await user.click(search);
    await user.paste('quality');
    await user.keyboard('{Enter}');
    expect(screen.getByTestId('compiled-search')).toHaveTextContent("attributes.run_name ILIKE '%quality%'");
    expect(search).toHaveValue('quality');

    await user.click(search);
    expect(screen.getByText('Recent')).toBeInTheDocument();
    await user.click(screen.getByText(codeText('quality')));
    expect(screen.getByTestId('compiled-search')).toHaveTextContent("attributes.run_name ILIKE '%quality%'");
    expect(search).toHaveValue('quality');

    await user.click(search);
    await user.clear(search);
    await user.keyboard('{Enter}');
    expect(screen.getByTestId('compiled-search')).toBeEmptyDOMElement();
    expect(search).toHaveValue('');
  });
});
