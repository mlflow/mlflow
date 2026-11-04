import { createRef } from 'react';
import { describe, it, expect, jest } from '@jest/globals';
import userEvent from '@testing-library/user-event';

import { renderWithDesignSystem, screen, act } from '../../../common/utils/TestUtils.react18';
import type { ExperimentRunsSelectorResult } from '../../components/experiment-page/utils/experimentRuns.selector';
import { EvalRunsFilterButton, type EvalRunsFilterButtonHandle } from './EvalRunsFilterButton';
import type { EvalRunsFilterClause } from './evalRunsFilterFields';
import { EvalRunsFilterOperator } from './evalRunsFilterFields';

const makeRunsData = (overrides: Partial<ExperimentRunsSelectorResult> = {}): ExperimentRunsSelectorResult => ({
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
  ...overrides,
});

const runNameClause: EvalRunsFilterClause = {
  field: 'attributes.run_name',
  operator: EvalRunsFilterOperator.EQUALS,
  value: 'foo',
};

// Open a SimpleSelect (by its accessible name) and click the option with the given visible label.
// SimpleSelect only mounts its options once opened, mirroring the CompareRunContour axis-picker helper.
const selectOption = async (comboboxName: string, optionText: string) => {
  await userEvent.click(screen.getByRole('combobox', { name: comboboxName }));
  const option = screen.getAllByRole('option').find((candidate) => candidate.textContent === optionText);
  if (!option) {
    throw new Error(`Option "${optionText}" not found in "${comboboxName}" select`);
  }
  await userEvent.click(option);
};

describe('EvalRunsFilterButton', () => {
  // @ui-test-skill-generated
  it('commits only the complete clauses on Apply and closes the popover', async () => {
    const setClauses = jest.fn();
    renderWithDesignSystem(<EvalRunsFilterButton clauses={[]} setClauses={setClauses} runsData={makeRunsData()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
    await selectOption('Filter field', 'Run name');
    await userEvent.click(screen.getByRole('textbox', { name: 'Filter value' }));
    await userEvent.paste('foo');

    // Add a second, empty row that must be dropped on Apply.
    await userEvent.click(screen.getByRole('button', { name: 'Add filter' }));
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));

    expect(setClauses).toHaveBeenCalledWith([
      { field: 'attributes.run_name', operator: EvalRunsFilterOperator.EQUALS, value: 'foo' },
    ]);
    // The popover closed, so its field pickers are gone.
    expect(screen.queryByRole('combobox', { name: 'Filter field' })).not.toBeInTheDocument();
  });

  it('shows an icon-only trigger when inactive and a labelled count when a filter is set', () => {
    const { rerender } = renderWithDesignSystem(
      <EvalRunsFilterButton clauses={[]} setClauses={jest.fn()} runsData={makeRunsData()} />,
    );

    expect(screen.getByRole('button', { name: 'Filters' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Filters \(/ })).not.toBeInTheDocument();

    rerender(<EvalRunsFilterButton clauses={[runNameClause]} setClauses={jest.fn()} runsData={makeRunsData()} />);

    expect(screen.getByRole('button', { name: /Filters \(1\)/ })).toBeInTheDocument();
  });

  it('resets the operator to the first valid one and clears the value when the field changes', async () => {
    renderWithDesignSystem(<EvalRunsFilterButton clauses={[]} setClauses={jest.fn()} runsData={makeRunsData()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));

    // Start with a numeric field, pick a non-default operator, and enter a value.
    await selectOption('Filter field', 'Start time (epoch ms)');
    await selectOption('Filter operator', '>');
    const numericInput = screen.getByRole('textbox', { name: 'Filter value' });
    expect(numericInput).toHaveAttribute('inputmode', 'decimal');
    await userEvent.click(numericInput);
    await userEvent.paste('5');

    // Switching to a string field resets the operator to '=' and clears the value.
    await selectOption('Filter field', 'Run name');
    expect(screen.getByRole('combobox', { name: 'Filter operator' })).toHaveTextContent('=');
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toHaveValue('');
  });

  it('renders a value dropdown for an enum field and an editable typeahead for a string field', async () => {
    renderWithDesignSystem(<EvalRunsFilterButton clauses={[]} setClauses={jest.fn()} runsData={makeRunsData()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));

    await selectOption('Filter field', 'Status');
    expect(screen.getByRole('combobox', { name: 'Filter value' })).toBeInTheDocument();
    expect(screen.queryByRole('textbox', { name: 'Filter value' })).not.toBeInTheDocument();

    await selectOption('Filter field', 'Run name');
    expect(screen.getByRole('textbox', { name: 'Filter value' })).toBeInTheDocument();
  });

  it.each([
    {
      excludeIssueDetectionRuns: false,
      expectedOptions: ['Evaluation', 'Agent evaluate', 'Issue detection'],
    },
    {
      excludeIssueDetectionRuns: true,
      expectedOptions: ['Evaluation', 'Agent evaluate'],
    },
  ])(
    'shows the allowed run type options when issue detection exclusion is $excludeIssueDetectionRuns',
    async ({ excludeIssueDetectionRuns, expectedOptions }) => {
      renderWithDesignSystem(
        <EvalRunsFilterButton
          clauses={[]}
          setClauses={jest.fn()}
          runsData={makeRunsData()}
          excludeIssueDetectionRuns={excludeIssueDetectionRuns}
        />,
      );

      await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
      await selectOption('Filter field', 'Run type');
      await userEvent.click(screen.getByRole('combobox', { name: 'Filter value' }));

      expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(expectedOptions);
    },
  );

  it('suggests distinct loaded values while keeping the value input editable', async () => {
    const setClauses = jest.fn();
    renderWithDesignSystem(
      <EvalRunsFilterButton
        clauses={[]}
        setClauses={setClauses}
        runsData={makeRunsData({
          runInfos: [
            {
              artifactUri: '',
              endTime: 0,
              experimentId: 'exp-1',
              lifecycleStage: 'active',
              runUuid: 'run-1',
              runName: 'validation',
              startTime: 0,
              status: 'FINISHED',
            },
            {
              artifactUri: '',
              endTime: 0,
              experimentId: 'exp-1',
              lifecycleStage: 'active',
              runUuid: 'run-2',
              runName: 'training',
              startTime: 0,
              status: 'FINISHED',
            },
            {
              artifactUri: '',
              endTime: 0,
              experimentId: 'exp-1',
              lifecycleStage: 'active',
              runUuid: 'run-3',
              runName: 'validation',
              startTime: 0,
              status: 'FINISHED',
            },
          ],
        })}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
    await selectOption('Filter field', 'Run name');
    const input = screen.getByRole('textbox', { name: 'Filter value' });
    await userEvent.click(input);

    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['training', 'validation']);

    await userEvent.click(screen.getByRole('option', { name: 'validation' }));
    expect(input).toHaveValue('validation');

    await userEvent.clear(input);
    await userEvent.paste('custom');
    expect(input).toHaveValue('custom');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(setClauses).toHaveBeenCalledWith([
      { field: 'attributes.run_name', operator: EvalRunsFilterOperator.EQUALS, value: 'custom' },
    ]);
  });

  it('suggests distinct metric values while retaining numeric input validation', async () => {
    renderWithDesignSystem(
      <EvalRunsFilterButton
        clauses={[]}
        setClauses={jest.fn()}
        runsData={makeRunsData({
          metricKeyList: ['accuracy'],
          metricsList: [
            [{ key: 'accuracy', step: 0, timestamp: 0, value: 0.9 }],
            [{ key: 'accuracy', step: 0, timestamp: 0, value: 0.8 }],
            [{ key: 'accuracy', step: 0, timestamp: 0, value: 0.9 }],
          ],
        })}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
    await selectOption('Filter field', 'accuracy');
    const input = screen.getByRole('textbox', { name: 'Filter value' });
    expect(input).toHaveAttribute('inputmode', 'decimal');
    await userEvent.click(input);

    expect(screen.getAllByRole('option').map((option) => option.textContent)).toEqual(['0.8', '0.9']);

    await userEvent.paste('invalid');
    expect(input).toHaveValue('');
    await userEvent.paste('0.8');
    expect(input).toHaveValue('0.8');
  });

  it('keeps incomplete numeric values in the editor until they are valid', async () => {
    const setClauses = jest.fn();
    renderWithDesignSystem(<EvalRunsFilterButton clauses={[]} setClauses={setClauses} runsData={makeRunsData()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
    await selectOption('Filter field', 'Start time (epoch ms)');
    const input = screen.getByRole('textbox', { name: 'Filter value' });
    await userEvent.click(input);
    await userEvent.paste('1e');
    expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
    expect(setClauses).not.toHaveBeenCalled();

    await userEvent.paste('2');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(setClauses).toHaveBeenCalledWith([
      { field: 'attributes.start_time', operator: EvalRunsFilterOperator.EQUALS, value: '1e2' },
    ]);
  });

  it('appends a clause row when Add filter is clicked', async () => {
    renderWithDesignSystem(<EvalRunsFilterButton clauses={[]} setClauses={jest.fn()} runsData={makeRunsData()} />);

    await userEvent.click(screen.getByRole('button', { name: 'Filters' }));
    expect(screen.getAllByRole('combobox', { name: 'Filter field' })).toHaveLength(1);

    await userEvent.click(screen.getByRole('button', { name: 'Add filter' }));
    expect(screen.getAllByRole('combobox', { name: 'Filter field' })).toHaveLength(2);
  });

  it('opens for a column field while keeping active clauses staged until Apply', async () => {
    const filterButtonRef = createRef<EvalRunsFilterButtonHandle>();
    const setClauses = jest.fn();
    renderWithDesignSystem(
      <EvalRunsFilterButton
        clauses={[runNameClause]}
        setClauses={setClauses}
        runsData={makeRunsData({ metricKeyList: ['accuracy'] })}
        filterButtonRef={filterButtonRef}
      />,
    );

    act(() => filterButtonRef.current?.openForField('metrics.`accuracy`'));
    const fields = await screen.findAllByRole('combobox', { name: 'Filter field' });
    expect(fields).toHaveLength(2);
    expect(fields.find((field) => field.textContent === 'Run name')).toBeInTheDocument();
    expect(fields.find((field) => field.textContent === 'accuracy')).toBeInTheDocument();
    expect(setClauses).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));

    expect(setClauses).toHaveBeenCalledWith([runNameClause]);
    expect(screen.queryByRole('combobox', { name: 'Filter field' })).not.toBeInTheDocument();
  });
  // @ui-test-skill-generated
  it.each([
    ['metrics.`new-key`', 'textbox', '42'],
    ['params.`new-key`', 'textbox', 'value'],
    ['tags.`new-key`', 'textbox', 'value'],
  ])('edits a requested field absent from loaded data: %s', async (field, role, value) => {
    const filterButtonRef = createRef<EvalRunsFilterButtonHandle>();
    const setClauses = jest.fn();
    renderWithDesignSystem(
      <EvalRunsFilterButton
        clauses={[]}
        setClauses={setClauses}
        runsData={makeRunsData()}
        filterButtonRef={filterButtonRef}
      />,
    );
    act(() => filterButtonRef.current?.openForField(field));
    await userEvent.click(await screen.findByRole(role, { name: 'Filter value' }));
    await userEvent.paste(value);
    expect(screen.getByRole('combobox', { name: 'Filter field' })).toHaveTextContent('new-key');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(setClauses).toHaveBeenCalledWith([{ field, operator: EvalRunsFilterOperator.EQUALS, value }]);
  });

  // @ui-test-skill-generated
  it('preserves an open draft through a second request and metadata refresh', async () => {
    const filterButtonRef = createRef<EvalRunsFilterButtonHandle>();
    const setClauses = jest.fn();
    const props = { clauses: [], setClauses, runsData: makeRunsData(), filterButtonRef };
    const { rerender } = renderWithDesignSystem(<EvalRunsFilterButton {...props} />);
    act(() => filterButtonRef.current?.openForField('attributes.run_name'));
    await userEvent.click(await screen.findByRole('textbox', { name: 'Filter value' }));
    await userEvent.paste('draft');
    act(() => filterButtonRef.current?.openForField('metrics.`accuracy`'));
    await screen.findAllByRole('textbox', { name: 'Filter value' });
    rerender(<EvalRunsFilterButton {...props} runsData={makeRunsData()} />);
    const valueInputs = screen.getAllByRole('textbox', { name: 'Filter value' });
    expect(valueInputs).toHaveLength(2);
    expect(valueInputs[0]).toHaveValue('draft');
    await userEvent.click(screen.getByRole('button', { name: 'Apply' }));
    expect(setClauses).toHaveBeenCalledWith([{ ...runNameClause, value: 'draft' }]);
    rerender(<EvalRunsFilterButton {...props} runsData={makeRunsData()} />);
    expect(screen.queryByRole('textbox', { name: 'Filter value' })).not.toBeInTheDocument();
  });
});
