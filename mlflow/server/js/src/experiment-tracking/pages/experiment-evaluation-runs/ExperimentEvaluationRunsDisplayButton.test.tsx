import { describe, expect, jest, test } from '@jest/globals';
import { DesignSystemProvider } from '@databricks/design-system';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { RunEntity } from '../../types';
import { ExperimentEvaluationRunsDisplayButton, parseSortValue } from './ExperimentEvaluationRunsDisplayButton';
import { ExperimentEvaluationRunsPageMode } from './hooks/useExperimentEvaluationRunsPageMode';

const { history } = setupTestRouter();

const makeRun = (tags: RunEntity['data']['tags'] = []): RunEntity => ({
  data: {
    params: [{ key: 'model', value: 'gpt-4o' }],
    tags,
    metrics: [],
  },
  info: {
    runUuid: 'run-1',
    runName: 'Run 1',
    startTime: 1,
    artifactUri: '',
    endTime: 0,
    experimentId: 'exp-1',
    lifecycleStage: 'active',
    status: 'FINISHED',
  },
  inputs: {
    datasetInputs: [
      {
        dataset: { name: 'eval-set', digest: 'digest-1', profile: '', schema: '', source: '', sourceType: '' },
        tags: [],
      },
    ],
  },
});

const renderDisplay = (props: Partial<ComponentProps<typeof ExperimentEvaluationRunsDisplayButton>> = {}) => {
  const setGroupByConfig = jest.fn();
  const setSorting = jest.fn();
  const setViewMode = jest.fn();
  render(
    <TestRouter
      history={history}
      initialEntries={['/p']}
      routes={[
        testRoute(
          <IntlProvider locale="en">
            <DesignSystemProvider>
              <ExperimentEvaluationRunsDisplayButton
                runs={[makeRun()]}
                selectedColumns={{ run_name: true, created_at: true }}
                setSelectedColumns={jest.fn()}
                groupByConfig={null}
                setGroupByConfig={setGroupByConfig}
                sorting={[]}
                setSorting={setSorting}
                viewMode={ExperimentEvaluationRunsPageMode.TRACES}
                setViewMode={setViewMode}
                {...props}
              />
            </DesignSystemProvider>
          </IntlProvider>,
        ),
      ]}
    />,
  );
  return { setGroupByConfig, setSorting, setViewMode };
};

describe('ExperimentEvaluationRunsDisplayButton', () => {
  test('parses sort values containing keyed column separators', () => {
    expect(parseSortValue('metric:quality:desc')).toEqual(['metric:quality', true]);
    expect(parseSortValue('created_at:asc')).toEqual(['created_at', false]);
  });

  test('keeps sorting and grouping in the display menu', async () => {
    const user = userEvent.setup();
    const { setGroupByConfig, setSorting } = renderDisplay();

    await user.click(await screen.findByRole('button', { name: 'Display' }));
    await user.click(screen.getByRole('menuitem', { name: /^Sort/ }));
    fireEvent.click(
      await screen.findByRole('menuitemradio', {
        name: 'Created at (ascending)',
      }),
    );
    expect(setSorting).toHaveBeenCalledWith([{ id: 'created_at', desc: false }]);

    await user.click(await screen.findByRole('button', { name: 'Display' }));
    await user.click(screen.getByRole('menuitem', { name: /^Group by/ }));
    fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'Dataset' }));
    expect(setGroupByConfig).toHaveBeenCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ mode: 'dataset', groupByData: 'dataset' }],
    });
  });

  test('offers lineage group-by options only for runs that have lineage tags', async () => {
    const user = userEvent.setup();
    const { setGroupByConfig } = renderDisplay({
      runs: [
        makeRun([
          { key: 'mlflow.genaiEvaluate.scorersDigest', value: '{"digest":"sha256:a","hashed":1,"total":1}' },
          { key: 'mlflow.genaiEvaluate.agent.function', value: 'pkg.predict' },
        ]),
      ],
    });

    await user.click(await screen.findByRole('button', { name: 'Display' }));
    await user.click(screen.getByRole('menuitem', { name: /^Group by/ }));
    expect(await screen.findByRole('menuitemcheckbox', { name: 'Agent' })).toBeInTheDocument();
    expect(
      screen.queryByRole('menuitemcheckbox', { name: 'mlflow.genaiEvaluate.scorersDigest' }),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('menuitemcheckbox', { name: 'Scorers' }));
    expect(setGroupByConfig).toHaveBeenCalledWith({
      aggregateFunction: 'average',
      groupByKeys: [{ mode: 'tag', groupByData: 'mlflow.genaiEvaluate.scorersDigest' }],
    });
  });

  test('hides lineage group-by options when no run has lineage tags', async () => {
    const user = userEvent.setup();
    renderDisplay();

    await user.click(await screen.findByRole('button', { name: 'Display' }));
    await user.click(screen.getByRole('menuitem', { name: /^Group by/ }));
    expect(await screen.findByRole('menuitemcheckbox', { name: 'Dataset' })).toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Scorers' })).not.toBeInTheDocument();
    expect(screen.queryByRole('menuitemcheckbox', { name: 'Agent' })).not.toBeInTheDocument();
  });

  test('offers the legacy charts view from the display menu', async () => {
    const user = userEvent.setup();
    const { setViewMode } = renderDisplay();

    await user.click(await screen.findByRole('button', { name: 'Display' }));
    fireEvent.click(await screen.findByRole('menuitem', { name: 'View legacy charts' }));

    expect(setViewMode).toHaveBeenCalledWith(ExperimentEvaluationRunsPageMode.CHARTS);
  });

  test('opens column visibility from the display menu', async () => {
    const user = userEvent.setup();
    const setSelectedColumns = jest.fn();
    renderDisplay({
      selectedColumns: { run_name: true, created_at: true, 'metric.accuracy': true },
      setSelectedColumns,
    });

    await user.click(await screen.findByRole('button', { name: 'Display' }));
    await user.click(screen.getByRole('menuitem', { name: 'Columns' }));
    fireEvent.click(await screen.findByRole('menuitemcheckbox', { name: 'accuracy' }));

    expect(setSelectedColumns).toHaveBeenCalledWith({
      run_name: true,
      created_at: true,
      'metric.accuracy': false,
    });
  });
});
