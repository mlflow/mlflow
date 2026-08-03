import { describe, test, expect } from '@jest/globals';

import { renderWithDesignSystem, screen } from '@mlflow/mlflow/src/common/utils/TestUtils.react18';
import { testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import type { KeyValueEntity } from '../../../common/types';
import { MLFLOW_RUN_TYPE_TAG, MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP, RunPageTabName } from '../../constants';
import { RunViewModeSwitch } from './RunViewModeSwitch';

const tags = (values: Record<string, string>): Record<string, KeyValueEntity> =>
  Object.fromEntries(Object.entries(values).map(([key, value]) => [key, { key, value }])) as Record<
    string,
    KeyValueEntity
  >;

const renderComponent = (props: Parameters<typeof RunViewModeSwitch>[0] = {}) =>
  renderWithDesignSystem(
    <TestRouter
      initialEntries={['/experiment/123/run/abc']}
      routes={[testRoute(<RunViewModeSwitch {...props} />, '/experiment/:experimentId/run/:runUuid')]}
    />,
  );

describe('RunViewModeSwitch', () => {
  test('hides the evaluation sweep tab on a run that is not a sweep', () => {
    renderComponent();
    expect(screen.getByRole('tab', { name: 'Overview' })).toBeInTheDocument();
    expect(screen.queryByRole('tab', { name: 'Evaluation sweep' })).not.toBeInTheDocument();

    renderComponent({ runTags: tags({ [MLFLOW_RUN_TYPE_TAG]: 'genai_evaluate' }) });
    expect(screen.queryByRole('tab', { name: 'Evaluation sweep' })).not.toBeInTheDocument();
  });

  test('shows the evaluation sweep tab on a sweep parent run', () => {
    renderComponent({ runTags: tags({ [MLFLOW_RUN_TYPE_TAG]: MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP }) });
    expect(screen.getByRole('tab', { name: 'Evaluation sweep' })).toBeInTheDocument();
  });

  test('appends the evaluation sweep tab to a custom tab list without duplicating it', () => {
    renderComponent({
      runTags: tags({ [MLFLOW_RUN_TYPE_TAG]: MLFLOW_RUN_TYPE_VALUE_GENAI_EVALUATE_SWEEP }),
      visibleTabs: [RunPageTabName.OVERVIEW, RunPageTabName.EVALUATION_SWEEP],
    });

    expect(screen.getAllByRole('tab')).toHaveLength(2);
    expect(screen.getAllByRole('tab', { name: 'Evaluation sweep' })).toHaveLength(1);
  });
});
