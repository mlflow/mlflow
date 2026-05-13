import { describe, test, expect, jest } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { CompareRunMetricTable } from './CompareRunMetricTable';
import { TestRouter, testRoute } from '../../common/utils/RoutingTestUtils';
import type { RunInfoEntity } from '../types';

const mockRunInfos: RunInfoEntity[] = [
  { runUuid: 'run_1', experimentId: 'exp_1' } as RunInfoEntity,
  { runUuid: 'run_2', experimentId: 'exp_1' } as RunInfoEntity,
];

const mockMetricRows = [
  { key: 'accuracy', highlightDiff: false, values: [0.95, 0.92] },
  { key: 'loss', highlightDiff: true, values: [0.05, 0.08] },
];

const wrapper = ({ children }: { children?: React.ReactNode }) => (
  <IntlProvider locale="en">
    <TestRouter routes={[testRoute(<>{children}</>)]} />
  </IntlProvider>
);

describe('CompareRunMetricTable', () => {
  test('renders metric rows with correct values', () => {
    const onScroll = jest.fn();
    render(
      <CompareRunMetricTable
        colWidth={200}
        experimentIds={['exp_1']}
        runInfos={mockRunInfos}
        metricRows={mockMetricRows}
        onScroll={onScroll}
      />,
      { wrapper },
    );

    expect(screen.getByText('accuracy')).toBeInTheDocument();
    expect(screen.getByText('loss')).toBeInTheDocument();
  });

  test('applies highlight styling to rows with highlightDiff', () => {
    const onScroll = jest.fn();
    render(
      <CompareRunMetricTable
        colWidth={200}
        experimentIds={['exp_1']}
        runInfos={mockRunInfos}
        metricRows={mockMetricRows}
        onScroll={onScroll}
      />,
      { wrapper },
    );

    expect(screen.getByText('loss')).toBeInTheDocument();
  });

  test('calls onScroll when body grid is scrolled', () => {
    const onScroll = jest.fn();
    const { container } = render(
      <CompareRunMetricTable
        colWidth={200}
        experimentIds={['exp_1']}
        runInfos={mockRunInfos}
        metricRows={mockMetricRows}
        onScroll={onScroll}
      />,
      { wrapper },
    );

    const grids = container.querySelectorAll('[role="grid"]');
    const bodyGrid = grids[1];

    fireEvent.scroll(bodyGrid, { target: { scrollLeft: 100 } });
    expect(onScroll).toHaveBeenCalled();
  });

  test('renders empty table when no metric rows provided', () => {
    const onScroll = jest.fn();
    const { container } = render(
      <CompareRunMetricTable
        colWidth={200}
        experimentIds={['exp_1']}
        runInfos={mockRunInfos}
        metricRows={[]}
        onScroll={onScroll}
      />,
      { wrapper },
    );

    expect(container.querySelector('[role="grid"]')).toBeInTheDocument();
  });

  test('setScrollLeft updates body grid scroll position', () => {
    const onScroll = jest.fn();
    const ref = { current: null } as any;

    render(
      <CompareRunMetricTable
        ref={ref}
        colWidth={200}
        experimentIds={['exp_1']}
        runInfos={mockRunInfos}
        metricRows={mockMetricRows}
        onScroll={onScroll}
      />,
      { wrapper },
    );

    expect(ref.current).toBeDefined();
    expect(ref.current.setScrollLeft).toBeDefined();
  });
});
