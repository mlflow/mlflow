import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ReviewQueueList } from './ReviewQueueList';
import type { ReviewQueueItem, ReviewStatus } from './types';

const mockTraces = jest.fn().mockReturnValue({ data: [] });
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  useGetTracesById: (...args: unknown[]) => mockTraces(...args),
}));

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

const item = (
  itemId: string,
  status: ReviewStatus,
  completedBy?: string,
  creationTimeMs = 1_780_000_000_000,
): ReviewQueueItem => ({
  queue_id: 'rq-1',
  item_type: 'TRACE',
  item_id: itemId,
  status,
  completed_by: completedBy,
  creation_time_ms: creationTimeMs,
  last_update_time_ms: creationTimeMs,
});

const NOW = 1_780_000_010_000;

describe('ReviewQueueList', () => {
  it('renders all column headers', () => {
    renderWithProviders(<ReviewQueueList items={[item('tr-1', 'PENDING')]} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.getByText('Request')).toBeInTheDocument();
    expect(screen.getByText('Response')).toBeInTheDocument();
    expect(screen.getByText('Status')).toBeInTheDocument();
    expect(screen.getByText('Date added')).toBeInTheDocument();
  });

  it('shows minute/second-fidelity relative time for the date added', () => {
    // Regression: a freshly-added trace used to render "1h ago" because the
    // formatter clamped its smallest unit to 1 hour.
    const items = [
      item('tr-just-now', 'PENDING', undefined, NOW - 10_000),
      item('tr-min', 'PENDING', undefined, NOW - 5 * 60_000),
      item('tr-hr', 'PENDING', undefined, NOW - 2 * 60 * 60_000),
      item('tr-day', 'PENDING', undefined, NOW - 3 * 24 * 60 * 60_000),
    ];
    renderWithProviders(<ReviewQueueList items={items} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.getByText('just now')).toBeInTheDocument();
    expect(screen.getByText('5m ago')).toBeInTheDocument();
    expect(screen.getByText('2h ago')).toBeInTheDocument();
    expect(screen.getByText('3d ago')).toBeInTheDocument();
  });

  it('renders status tags for all items in a flat list', () => {
    renderWithProviders(
      <ReviewQueueList
        items={[item('tr-1', 'PENDING'), item('tr-2', 'COMPLETE', 'bob')]}
        onOpen={jest.fn()}
        nowMs={NOW}
      />,
    );
    expect(screen.getByText('Needs review')).toBeInTheDocument();
    expect(screen.getByText('Reviewed')).toBeInTheDocument();
  });

  it('renders request and response previews from trace data', () => {
    mockTraces.mockReturnValueOnce({
      data: [{ info: { trace_id: 'tr-1', request_preview: 'Hello world', response_preview: 'Hi there' } }],
    });
    renderWithProviders(<ReviewQueueList items={[item('tr-1', 'PENDING')]} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.getByText('Hi there')).toBeInTheDocument();
  });

  it('renders an empty state when there are no traces', () => {
    renderWithProviders(<ReviewQueueList items={[]} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.getByText('No traces in this queue yet')).toBeInTheDocument();
  });

  it('shows an "Add traces" CTA in the empty state that calls onGoToTraces', () => {
    const onGoToTraces = jest.fn();
    renderWithProviders(<ReviewQueueList items={[]} onOpen={jest.fn()} nowMs={NOW} onGoToTraces={onGoToTraces} />);
    fireEvent.click(screen.getByText('Add traces'));
    expect(onGoToTraces).toHaveBeenCalledTimes(1);
  });

  it('omits the "Add traces" CTA when onGoToTraces is not provided', () => {
    renderWithProviders(<ReviewQueueList items={[]} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.queryByText('Add traces')).not.toBeInTheDocument();
  });

  it('calls onOpen when a row is clicked', () => {
    const onOpen = jest.fn();
    renderWithProviders(<ReviewQueueList items={[item('tr-1', 'PENDING')]} onOpen={onOpen} nowMs={NOW} />);
    fireEvent.click(screen.getByText('Needs review'));
    expect(onOpen).toHaveBeenCalledTimes(1);
    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ item_id: 'tr-1' }));
  });

  it('renders the declined status label', () => {
    renderWithProviders(<ReviewQueueList items={[item('tr-3', 'DECLINED', 'carol')]} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.getByText('Declined')).toBeInTheDocument();
  });

  it('filters to only pending items when Needs review is clicked', () => {
    renderWithProviders(
      <ReviewQueueList
        items={[item('tr-1', 'PENDING'), item('tr-2', 'COMPLETE', 'bob')]}
        onOpen={jest.fn()}
        nowMs={NOW}
      />,
    );
    expect(screen.getByText('Needs review')).toBeInTheDocument();
    expect(screen.getByText('Reviewed')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Needs review (1)'));
    expect(screen.getByText('Needs review')).toBeInTheDocument();
    expect(screen.queryByText('Reviewed')).not.toBeInTheDocument();
  });

  it('filters to only completed items when Completed is clicked', () => {
    renderWithProviders(
      <ReviewQueueList
        items={[item('tr-1', 'PENDING'), item('tr-2', 'COMPLETE', 'bob')]}
        onOpen={jest.fn()}
        nowMs={NOW}
      />,
    );
    fireEvent.click(screen.getByText('Completed (1)'));
    expect(screen.queryByText('Needs review')).not.toBeInTheDocument();
    expect(screen.getByText('Reviewed')).toBeInTheDocument();
  });

  it('sorts by Date added when the column header is clicked', () => {
    mockTraces.mockReturnValue({
      data: [
        { info: { trace_id: 'early', request_preview: 'req-early' } },
        { info: { trace_id: 'late', request_preview: 'req-late' } },
      ],
    });
    const items = [
      item('early', 'PENDING', undefined, 1_780_000_000_000),
      item('late', 'PENDING', undefined, 1_780_000_090_000),
    ];
    renderWithProviders(<ReviewQueueList items={items} onOpen={jest.fn()} nowMs={NOW + 100_000} />);
    const dateHeader = screen.getByText('Date added');
    fireEvent.click(dateHeader);
    const rows = screen.getAllByText(/req-early|req-late/);
    expect(rows[0].textContent).toBe('req-early');
    expect(rows[1].textContent).toBe('req-late');
    fireEvent.click(dateHeader);
    const rowsDesc = screen.getAllByText(/req-early|req-late/);
    expect(rowsDesc[0].textContent).toBe('req-late');
    expect(rowsDesc[1].textContent).toBe('req-early');
    mockTraces.mockReturnValue({ data: [] });
  });

  it('clears the sort on the third Date added click (asc -> desc -> none)', () => {
    mockTraces.mockReturnValue({
      data: [
        { info: { trace_id: 'early', request_preview: 'req-early' } },
        { info: { trace_id: 'late', request_preview: 'req-late' } },
      ],
    });
    // Provided late-first; with no sort the rows keep this input order.
    const items = [
      item('late', 'PENDING', undefined, 1_780_000_090_000),
      item('early', 'PENDING', undefined, 1_780_000_000_000),
    ];
    renderWithProviders(<ReviewQueueList items={items} onOpen={jest.fn()} nowMs={NOW + 100_000} />);
    const dateHeader = screen.getByText('Date added');
    fireEvent.click(dateHeader); // asc
    fireEvent.click(dateHeader); // desc
    fireEvent.click(dateHeader); // none -> back to input order
    const rows = screen.getAllByText(/req-early|req-late/);
    expect(rows[0].textContent).toBe('req-late');
    expect(rows[1].textContent).toBe('req-early');
    mockTraces.mockReturnValue({ data: [] });
  });

  it('shows an empty state when the active filter matches no traces', () => {
    renderWithProviders(<ReviewQueueList items={[item('tr-1', 'PENDING')]} onOpen={jest.fn()} nowMs={NOW} />);
    // No completed items, so the "Completed (0)" bucket renders the empty message.
    fireEvent.click(screen.getByText('Completed (0)'));
    expect(screen.getByText('No traces match this filter.')).toBeInTheDocument();
  });

  it('clears row selection when the status filter changes', () => {
    renderWithProviders(
      <ReviewQueueList
        items={[item('tr-1', 'PENDING'), item('tr-2', 'COMPLETE', 'bob')]}
        onOpen={jest.fn()}
        nowMs={NOW}
        onRemoveItems={jest.fn()}
      />,
    );
    // checkboxes[0] is select-all; [1] is the first row.
    const checkboxes = screen.getAllByRole('checkbox');
    fireEvent.click(checkboxes[1]);
    expect(screen.getByRole('button', { name: 'Unassign' })).toBeInTheDocument();
    // Switching filters resets the selection so hidden rows can't be deleted.
    fireEvent.click(screen.getByText('Completed (1)'));
    expect(screen.queryByRole('button', { name: 'Unassign' })).not.toBeInTheDocument();
  });
});
