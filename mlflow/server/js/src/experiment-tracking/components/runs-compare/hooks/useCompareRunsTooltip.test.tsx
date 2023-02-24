import { mount, ReactWrapper } from 'enzyme';
import React from 'react';
import {
  CompareRunsTooltipBodyComponent,
  CompareRunsTooltipWrapper,
  useCompareRunsTooltip,
} from './useCompareRunsTooltip';

const defaultBodyComponent: CompareRunsTooltipBodyComponent = ({ runUuid }) => (
  <div data-testid='tooltip-body'>
    tooltip body
    <div data-testid='tooltip-body-run-uuid'>{runUuid}</div>
  </div>
);

const createWrapper = (
  contextData: string | undefined = undefined,
  hoverData: string | undefined = undefined,
  TooltipBodyComponent: CompareRunsTooltipBodyComponent = defaultBodyComponent,
) => {
  const ContextComponent = ({ children }: React.PropsWithChildren<any>) => (
    <CompareRunsTooltipWrapper contextData={contextData} component={TooltipBodyComponent}>
      {children}
    </CompareRunsTooltipWrapper>
  );
  const TargetComponent = () => {
    const { setTooltip, resetTooltip } = useCompareRunsTooltip(hoverData);

    return (
      <button
        data-testid='fake-chart'
        onMouseEnter={({ runId, nativeEvent, additionalAxisData }: any) =>
          setTooltip(runId, nativeEvent, additionalAxisData)
        }
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

describe('useCompareRunsTooltip', () => {
  let wrapper: ReactWrapper;

  beforeEach(() => {});

  const getFakeChart = () => wrapper.find('[data-testid="fake-chart"]');
  const getTooltipArea = () => wrapper.find(CompareRunsTooltipWrapper);
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
    wrapper = createWrapper('context-data', 'hover-data', ({ runUuid, contextData, hoverData }) => (
      <div data-testid='tooltip-body'>
        {runUuid},{contextData},{hoverData}
      </div>
    ));
    const fakeChart = getFakeChart();
    fakeChart.simulate('mouseenter', { runId: 'fake_run_id_123' });

    expect(getTooltip().exists()).toBe(true);

    expect(getTooltip().html()).toContain('fake_run_id_123,context-data,hover-data');
  });

  it('properly sets additional X axis data', () => {
    wrapper = createWrapper(undefined, undefined, ({ runUuid, additionalAxisData }) => (
      <div data-testid='tooltip-body'>
        {runUuid}, step: {additionalAxisData.step}
      </div>
    ));
    const fakeChart = getFakeChart();
    fakeChart.simulate('mouseenter', {
      runId: 'fake_run_id_123',
      additionalAxisData: { step: 5 },
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
      <div data-testid='tooltip-body'>{!isHovering && <div>context menu only content</div>}</div>
    ));
    // Hover over the run
    getFakeChart().simulate('mouseenter', { runId: 'fake_run_id_123' });
    expect(getTooltip().exists()).toBe(true);
    expect(getTooltip().html()).not.toContain('context menu only content');

    // Lower the left mouse button
    getTooltipArea().simulate('mousedown', { button: 0 });

    // Next raise it without moving the cursor
    getTooltipArea().simulate('click');

    // Assert that the resulting context-menu-only content is present
    expect(getTooltip().html()).toContain('context menu only content');
  });

  it('properly prevents the context menu from appearing if user has moved the mouse', () => {
    wrapper = createWrapper(undefined, undefined, ({ isHovering }) => (
      <div data-testid='tooltip-body'>{!isHovering && <div>context menu only content</div>}</div>
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
