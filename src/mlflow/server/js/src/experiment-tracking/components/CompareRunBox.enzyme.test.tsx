/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { LegacySelect } from '@databricks/design-system';
import { CompareRunBox } from './CompareRunBox';
import { RunInfo } from '../sdk/MlflowMessages';
import { mountWithIntl } from 'common/utils/TestUtils.enzyme';
import { LazyPlot } from './LazyPlot';

describe('CompareRunBox', () => {
  let wrapper;
  const runUuids = ['1', '2', '3'];
  const commonProps = {
    runUuids,
    runInfos: runUuids.map((run_uuid) =>
      (RunInfo as any).fromJs({
        run_uuid,
        experiment_id: '0',
      }),
    ),
    runDisplayNames: runUuids,
  };
  test('should render with minimal props without exploding', () => {
    const props = {
      ...commonProps,
      paramLists: [[{ key: 'param', value: 1 }], [{ key: 'param', value: 2 }], [{ key: 'param', value: 3 }]],
      metricLists: [[{ key: 'metric', value: 4 }], [{ key: 'metric', value: 5 }], [{ key: 'metric', value: 6 }]],
    };
    wrapper = mountWithIntl(<CompareRunBox {...props} />);
    expect(wrapper.find(LazyPlot).isEmpty()).toBe(true);
    expect(wrapper.text()).toContain('Select parameters/metrics to plot.');
    const selectors = wrapper.find(LegacySelect);
    expect(selectors.length).toBe(2);
    // Set x-axis to 'param'
    const xAxisSelector = selectors.at(0);
    xAxisSelector.find('input[type="search"]').simulate('mouseDown');
    // `wrapper.find` can't find the selector options because they appear in the top level of the
    // document.
    (document.querySelectorAll('[data-test-id="axis-option"]')[0] as any).click();
    expect(xAxisSelector.text()).toContain('param');
    // Set y-axis to 'metric'
    const yAxisSelector = selectors.at(1);
    yAxisSelector.find('input[type="search"]').simulate('mouseDown');
    (document.querySelectorAll('[data-test-id="axis-option"]')[3] as any).click();
    expect(yAxisSelector.text()).toContain('metric');
    wrapper.update();
    expect(wrapper.find(LazyPlot).exists()).toBe(true);
    expect(wrapper.text()).not.toContain('Select parameters/metrics to plot.');
  });
});
