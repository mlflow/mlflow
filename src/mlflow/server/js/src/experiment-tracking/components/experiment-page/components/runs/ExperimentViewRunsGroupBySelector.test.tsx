import { IntlProvider } from 'react-intl';
import { render, screen } from '../../../../../common/utils/TestUtils.react18';
import { ExperimentRunsSelectorResult } from '../../utils/experimentRuns.selector';
import { ExperimentViewRunsGroupBySelector } from './ExperimentViewRunsGroupBySelector';
import userEventGlobal, { PointerEventsCheckLevel } from '@testing-library/user-event-14';
import { DesignSystemProvider } from '@databricks/design-system';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

describe('ExperimentViewRunsGroupBySelector', () => {
  const runsDataDatasets: Partial<ExperimentRunsSelectorResult> = {
    datasetsList: [
      [
        {
          dataset: { digest: 'abcdef', name: 'eval-dataset', profile: '', schema: '', source: '', source_type: '' },
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

  const renderComponent = (initialGroupBy = '', runsData = defaultRunsData, onChange = jest.fn()) => {
    return render(
      <ExperimentViewRunsGroupBySelector
        groupBy={initialGroupBy}
        isLoading={false}
        onChange={onChange}
        runsData={runsData}
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );
  };
  test('displays selector for dataset, tags and params with nothing checked', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));

    expect(screen.getByRole('menuitemradio', { name: 'tag1' })).not.toBeChecked();
    expect(screen.getByRole('menuitemradio', { name: 'tag2' })).not.toBeChecked();
    expect(screen.getByRole('menuitemradio', { name: 'param1' })).not.toBeChecked();
    expect(screen.getByRole('menuitemradio', { name: 'param2' })).not.toBeChecked();
    expect(screen.getByRole('menuitemradio', { name: 'Dataset' })).not.toBeChecked();
  });

  test('displays selector for dataset, tags and params with group by already set', async () => {
    renderComponent('param.min.param1');

    await userEvent.click(screen.getByRole('button', { name: /^Group:/ }));

    expect(screen.getByRole('menuitemradio', { name: 'param1' })).toBeChecked();
  });

  test('displays selector with no datasets present', async () => {
    renderComponent('', {
      ...defaultRunsData,
      datasetsList: [],
    });

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));

    expect(screen.queryByRole('menuitemradio', { name: 'Dataset' })).not.toBeInTheDocument();
  });

  test('selects group by tag option', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByText('Group by'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'tag1' }));

    expect(onChange).toBeCalledWith('tag.average.tag1');
  });

  test('selects group by parameter option', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'param2' }));

    expect(onChange).toBeCalledWith('param.average.param2');
  });

  test('selects group by dataset option', async () => {
    const onChange = jest.fn();
    renderComponent(undefined, undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Dataset' }));

    expect(onChange).toBeCalledWith('dataset.average.dataset');
  });

  test('changes aggregation function', async () => {
    const onChange = jest.fn();
    renderComponent('param.min.param1', undefined, onChange);

    await userEvent.click(screen.getByRole('button', { name: /^Group:/ }));
    await userEvent.click(screen.getByLabelText('Change aggregation function'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Maximum' }));

    expect(onChange).toHaveBeenLastCalledWith('param.max.param1');

    await userEvent.click(screen.getByLabelText('Change aggregation function'));
    await userEvent.click(screen.getByRole('menuitemradio', { name: 'Average' }));

    expect(onChange).toHaveBeenLastCalledWith('param.average.param1');
  });

  test('filters by param name', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.type(screen.getByRole('textbox'), 'param2');

    expect(screen.queryByRole('menuitemradio', { name: 'param2' })).toBeInTheDocument();

    expect(screen.queryByRole('menuitemradio', { name: 'Dataset' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemradio', { name: 'param1' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemradio', { name: 'tag1' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemradio', { name: 'tag2' })).not.toBeInTheDocument();
  });

  test('cannot change aggregation function when grouping is disabled', async () => {
    renderComponent();

    await userEvent.click(screen.getByRole('button', { name: /Group by/ }));
    await userEvent.click(screen.getByLabelText('Change aggregation function'));

    expect(screen.getByRole('menuitemradio', { name: 'Maximum' })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitemradio', { name: 'Minimum' })).toHaveAttribute('aria-disabled', 'true');
    expect(screen.getByRole('menuitemradio', { name: 'Average' })).toHaveAttribute('aria-disabled', 'true');
  });
});
