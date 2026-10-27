import { jest, describe, test, expect } from '@jest/globals';
import { useRef, useState } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ExperimentEvaluationRunsTable } from './ExperimentEvaluationRunsTable';
import { ExperimentEvaluationRunsRowVisibilityProvider } from './hooks/useExperimentEvaluationRunsRowVisibility';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';
import {
  EvalRunsTableKeyedColumnPrefix,
  getEvalRunsBaseSelectionState,
} from './ExperimentEvaluationRunsTable.constants';
import { createEvalRunsTableKeyedColumnKey } from './ExperimentEvaluationRunsTable.utils';
import { EvalRunsFilterButton, type EvalRunsFilterButtonHandle } from './EvalRunsFilterButton';
import type { EvalRunsFilterClause } from './evalRunsFilterFields';
import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';

// RunNameCell pulls run colors from Redux; stub the hooks so we don't need a Provider.
jest.mock('../../components/experiment-page/hooks/useExperimentRunColor', () => ({
  useGetExperimentRunColor: () => () => '#000000',
  useSaveExperimentRunColor: () => jest.fn(),
}));

const metricColumn = createEvalRunsTableKeyedColumnKey(EvalRunsTableKeyedColumnPrefix.METRIC, 'accuracy');

const filterRunsData: ExperimentRunsSelectorResult = {
  runInfos: [],
  runUuidsMatchingFilter: [],
  metricKeyList: ['accuracy'],
  paramKeyList: [],
  metricsList: [],
  paramsList: [],
  tagsList: [],
  modelVersionsByRunUuid: {},
  experimentTags: {},
  datasetsList: [],
};

const renderTable = (onFilterByColumn?: (field: string) => void) => (
  <ExperimentEvaluationRunsTable
    data={[]}
    uniqueColumns={[metricColumn]}
    selectedColumns={{ ...getEvalRunsBaseSelectionState(true), [metricColumn]: true }}
    setSelectedColumns={jest.fn()}
    setSelectedRunUuid={jest.fn()}
    isLoading={false}
    hasNextPage={false}
    rowSelection={{}}
    setRowSelection={jest.fn()}
    setSelectedDatasetWithRun={jest.fn()}
    setIsDrawerOpen={jest.fn()}
    viewMode={ExperimentEvaluationRunsPageMode.TRACES}
    enableNewEvalRunsUi
    onFilterByColumn={onFilterByColumn}
  />
);

const EvalRunsColumnFilteringHarness = () => {
  const filterButtonRef = useRef<EvalRunsFilterButtonHandle>(null);
  const [clauses, setClauses] = useState<EvalRunsFilterClause[]>([]);

  return (
    <>
      {renderTable((field) => filterButtonRef.current?.openForField(field))}
      <EvalRunsFilterButton
        clauses={clauses}
        setClauses={setClauses}
        runsData={filterRunsData}
        filterButtonRef={filterButtonRef}
      />
    </>
  );
};

const renderWithProviders = (children: React.ReactNode) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ExperimentEvaluationRunsRowVisibilityProvider>{children}</ExperimentEvaluationRunsRowVisibilityProvider>
      </DesignSystemProvider>
    </IntlProvider>,
  );

describe('ExperimentEvaluationRunsTable column filtering', () => {
  test('column filtering adds a menu action for supported keyed columns', async () => {
    const onFilterByColumn = jest.fn();
    renderWithProviders(renderTable(onFilterByColumn));

    await userEvent.click(screen.getByLabelText('Options for accuracy column'));
    expect(screen.getByRole('menuitem', { name: 'Filter by this column' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('menuitem', { name: 'Filter by this column' }));

    expect(onFilterByColumn).toHaveBeenCalledWith('metrics.`accuracy`');
  });

  test('has no filter action without a filter handler', async () => {
    renderWithProviders(renderTable());

    await userEvent.click(screen.getByLabelText('Options for accuracy column'));
    expect(screen.queryByRole('menuitem', { name: 'Filter by this column' })).not.toBeInTheDocument();
  });

  test('opens the filter popover with the selected field after the header menu closes', async () => {
    renderWithProviders(<EvalRunsColumnFilteringHarness />);

    await userEvent.click(screen.getByLabelText('Options for accuracy column'));
    await userEvent.click(screen.getByRole('menuitem', { name: 'Filter by this column' }));

    const fields = await screen.findAllByRole('combobox', { name: 'Filter field' });
    expect(fields).toHaveLength(1);
    expect(fields[0]).toHaveTextContent('accuracy');
  });
});
