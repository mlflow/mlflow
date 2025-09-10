import { IntlProvider } from 'react-intl';
import { render, screen } from '../../../../../common/utils/TestUtils.react18';
import type { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsGroupBySelector } from './ExperimentViewRunsGroupBySelector';
import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import type { RunsGroupByConfig } from '../../utils/experimentPage.group-row-utils';
import { RunGroupingAggregateFunction, RunGroupingMode } from '../../utils/experimentPage.row-types';
import { useState } from 'react';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

describe('ExperimentViewRunsGroupBySelector', () => {
  const runsDataDatasets: Partial<ExperimentRunsSelectorResult> = {
    datasetsList: [
      [
        {
          dataset: { digest: 'abcdef', name: 'eval-dataset', profile: '', schema: '', source: '', sourceType: '' },
          tags: [],
        },
      ],
    ],
  };
  const runsDataParams: Partial<ExperimentRunsSelectorResult> = {
    paramKeyList: ['param1', 'param2'],
  };
  const runsDataTags: Partial<ExperimentRunsSelectorResult> = {
    tagsList: [
      { tag1: { key: 'tag1', value: 'value1' } },
      { tag1: { key: 'tag1', value: 'value2' } },
      { tag2: { key: 'tag2', value: 'value2' } },
    ] as any,
  };
  const defaultRunsData: ExperimentRunsSelectorResult = {
    ...runsDataDatasets,
    ...runsDataParams,
    ...runsDataTags,
  } as any;

  const renderComponent = (
    initialGroupBy: RunsGroupByConfig | string | null = null,
    runsData = defaultRunsData,
    onChangeListener = jest.fn(),
  ) => {
    const TestComponent = () => {
      const [groupBy, setGroupBy] = useState<RunsGroupByConfig | string | null>(initialGroupBy);
      return (
        <ExperimentViewRunsGroupBySelector
          groupBy={groupBy}
          isLoading={false}
          runsData={runsData}
          useGroupedValuesInCharts
          onUseGroupedValuesInChartsChange={() => {}}
          onChange={(data) => {
            setGroupBy(data);
            onChangeListener(data);
          }}
        />
      );
    };
    return render(<TestComponent />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>{children}</DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  test('displays selector for dataset, tags and params with nothing checked', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));

    expect(screen.getByRole('menuitemcheckbox', { name: 'tag1' })).not.toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'tag2' })).not.toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'param1' })).not.toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'param2' })).not.toBeChecked();
    expect(screen.getByRole('menuitemcheckbox', { name: 'Dataset' })).not.toBeChecked();
  });

  test('displays selector for dataset, tags and params with group by already set (legacy group by key)', async () => {
    renderComponent('param.min.param1');

    await userEvent.click(screen.getByRole('button', { name: /^Group by:/ }));

    expect(screen.getByRole('menuitemcheckbox', { name: 'param1' })).toBeChecked();
  });

  test('displays selector for dataset, tags and params with group by already set', async () => {
    renderComponent({
      aggregateFunction: RunGroupingAggregateFunction.Min,
      groupByKeys: [
        {
          mode: RunGroupingMode.Param,
          groupByData: 'param1',
        },
      ],
    });

    await userEvent.click(screen.getByRole('button', { name: /^Group by:/ }));

    expect(screen.getByRole('menuitemcheckbox', { name: 'param1' })).toBeChecked();
  });

  test('displays selector with no datasets present', async () => {
    renderComponent(null, {
      ...defaultRunsData,
      datasetsList: [],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));

    expect(screen.queryByRole('menuitemcheckbox', { name: 'Dataset' })).not.toBeInTheDocument();
  });

  test('selects group by tag option', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByText('Group by'));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'tag1' }));

    expect(onChange).toHaveBeenCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ groupByData: 'tag1', mode: 'tag' }],
    });
  });

  test('selects group by parameter option', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'param2' }));

    expect(onChange).toHaveBeenCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ groupByData: 'param2', mode: 'param' }],
    });
  });

  test('selects group by dataset option', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Dataset' }));

    expect(onChange).toHaveBeenCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ groupByData: 'dataset', mode: 'dataset' }],
    });
  });

  test('changes aggregation function', async () => {
    const onChange = jest.fn();
    renderComponent('param.min.param1', undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /^Group by:/ }));
    await userEvent.click(screen.getByLabelText('Change aggregation function'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Maximum' }));

    expect(onChange).toHaveBeenLastCalledWith({
      aggregateFunction: 'max',
      groupByKeys: [{ groupByData: 'param1', mode: 'param' }],
    });

    await userEvent.click(screen.getByLabelText('Change aggregation function'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Average' }));

    expect(onChange).toHaveBeenLastCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ groupByData: 'param1', mode: 'param' }],
    });
  });

  test('filters by param name', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.type(screen.getByRole('textbox'), 'param2');

    expect(screen.queryByRole('menuitemcheckbox', { name: 'param2' })).toBeInTheDocument();

    expect(screen.queryByRole('menuitemcheckbox', { name: 'Dataset' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'param1' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'tag1' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'tag2' })).not.toBeInTheDocument();
  });

  test('cannot change aggregation function when grouping is disabled', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByLabelText('Change aggregation function'));

    expect(screen.getByRole('menuitemradio', { name: 'Maximum' })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitemradio', { name: 'Minimum' })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitemradio', { name: 'Average' })).toHaveAttribute('aria-disabled', 'true');
  });

  test('selects multiple group by keys and remove them one by one', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Dataset' }));

    expect(onChange).toHaveBeenCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ groupByData: 'dataset', mode: 'dataset' }],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'tag1' }));

    expect(onChange).toHaveBeenLastCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [
        { groupByData: 'dataset', mode: 'dataset' },
        { groupByData: 'tag1', mode: 'tag' },
      ],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'param2' }));

    expect(onChange).toHaveBeenLastCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [
        { groupByData: 'dataset', mode: 'dataset' },
        { groupByData: 'tag1', mode: 'tag' },
        { groupByData: 'param2', mode: 'param' },
      ],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'tag1' }));

    expect(onChange).toHaveBeenLastCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [
        { groupByData: 'dataset', mode: 'dataset' },
        { groupByData: 'param2', mode: 'param' },
      ],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Dataset' }));

    expect(onChange).toHaveBeenLastCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ groupByData: 'param2', mode: 'param' }],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'param2' }));

    expect(onChange).toHaveBeenLastCalledWith(null);
  });
});
