import type { ReactWrapper } from 'enzyme';
import { mount } from 'enzyme';
import React from 'react';
import type { RunsChartsTooltipBodyComponent } from './useRunsChartsTooltip';
import { RunsChartsTooltipWrapper, useRunsChartsTooltip } from './useRunsChartsTooltip';

const defaultBodyComponent: RunsChartsTooltipBodyComponent = ({ runUuid }) => (
  <div data-testid="tooltip-body">
    tooltip body
    <div data-testid="tooltip-body-run-uuid">{runUuid}</div>
  </div>
);

const createWrapper = (
  contextData: string | undefined = undefined,
  chartData: string | undefined = undefined,
  TooltipBodyComponent: RunsChartsTooltipBodyComponent = defaultBodyComponent,
) => {
  const ContextComponent = ({ children }: React.PropsWithChildren<any>) => (
    <RunsChartsTooltipWrapper contextData={contextData} component={TooltipBodyComponent}>
      {children}
    </RunsChartsTooltipWrapper>
  );
  const TargetComponent = () => {
    const { setTooltip, resetTooltip } = useRunsChartsTooltip(chartData);

    return (
      <button
        data-testid="fake-chart"
        onMouseEnter={({ runId, nativeEvent, hoverData }: any) => setTooltip(runId, nativeEvent, hoverData)}
        onMouseLeave={resetTooltip}
      />
    );
  };

  return mount(
    <ContextComponent>
      <TargetComponent />
    </ContextComponent>,
  );
};

describe('useRunsChartsTooltip', () => {
  // @ts-expect-error TS(2709): Cannot use namespace 'ReactWrapper' as a type.
  let wrapper: ReactWrapper;

  beforeEach(() => {});

  const getFakeChart = () => wrapper.find('[data-testid="fake-chart"]');
  const getTooltipArea = () => wrapper.find(RunsChartsTooltipWrapper);
  const getTooltipContainer = () => wrapper.find('[data-testid="tooltip-container"]');
  const getTooltip = () => wrapper.find('[data-testid="tooltip-body"]');

  it('properly shows and hides the tooltip', () => {
    wrapper = createWrapper();
    const fakeChart = getFakeChart();

    expect(getTooltip().exists()).toBe(false);

    fakeChart.simulate('mouseenter', { runId: 'fake_run_id_123' });
    wrapper.update();

    expect(getTooltip().exists()).toBe(true);

    fakeChart.simulate('mouseleave');

    expect(getTooltip().exists()).toBe(false);
  });

  it('properly sets the tooltip content', () => {
    wrapper = createWrapper('context-data', 'chart-data', ({ runUuid, contextData, chartData }) => (
      <div data-testid="tooltip-body">
        {runUuid},{contextData},{chartData}
      </div>
    ));
    const fakeChart = getFakeChart();
    fakeChart.simulate('mouseenter', { runId: 'fake_run_id_123' });

    expect(getTooltip().exists()).toBe(true);

    expect(getTooltip().html()).toContain('fake_run_id_123,context-data,chart-data');
  });

  it('properly sets hover data', () => {
    wrapper = createWrapper(undefined, undefined, ({ runUuid, hoverData }) => (
      <div data-testid="tooltip-body">
        {runUuid}, step: {hoverData.step}
      </div>
    ));
    const fakeChart = getFakeChart();
    fakeChart.simulate('mouseenter', {
      runId: 'fake_run_id_123',
      hoverData: { step: 5 },
    });

    expect(getTooltip().exists()).toBe(true);

    expect(getTooltip().html()).toContain('fake_run_id_123, step: 5');
  });

  it('properly sets the tooltip position', () => {
    wrapper = createWrapper();
    getFakeChart().simulate('mouseenter', { runId: 'fake_run_id_123' });
    getTooltipArea().simulate('mousemove', { nativeEvent: { offsetX: 400, offsetY: 200 } });

    const styles = getComputedStyle(getTooltipContainer().getDOMNode());

    expect(styles.transform).toBe(`translate3d(${400 + 1}px, ${200 + 1}px, 0)`);
  });

  it('properly invokes the context menu out of the tooltip', () => {
    wrapper = createWrapper(undefined, undefined, ({ isHovering }) => (
      <div data-testid="tooltip-body">{!isHovering && <div>context menu only content</div>}</div>
    ));
    // Hover over the run
    getFakeChart().simulate('mouseenter', { runId: 'fake_run_id_123' });
    expect(getTooltip().exists()).toBe(true);
    expect(getTooltip().html()).not.toContain('context menu only content');

    // Lower the left mouse button
    getTooltipArea().simulate('mousedown', { button: 0, pageX: 100, pageY: 100 });

    // Next raise it without moving the cursor
    getTooltipArea().simulate('click', { button: 0, pageX: 100, pageY: 100 });

    // Assert that the resulting context-menu-only content is present
    expect(getTooltip().html()).toContain('context menu only content');
  });

  it('properly prevents the context menu from appearing if user has moved the mouse', () => {
    wrapper = createWrapper(undefined, undefined, ({ isHovering }) => (
      <div data-testid="tooltip-body">{!isHovering && <div>context menu only content</div>}</div>
    ));
    // Hover over the run
    getFakeChart().simulate('mouseenter', { runId: 'fake_run_id_123' });

    // Lower the left mouse button
    getTooltipArea().simulate('mousedown', { button: 0 });

    // Wobble our cursor a little, then release the button
    getTooltipArea().simulate('mousemove');
    getTooltipArea().simulate('click');

    // Assert that the context menu content didn't appear
    expect(getTooltip().html()).not.toContain('context menu only content');
  });
});
