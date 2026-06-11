import { describe, jest, it, expect } from '@jest/globals';
import { render, screen, fireEvent } from '@testing-library/react';
import React from 'react';

import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from '@databricks/i18n';

import { ReviewQueueList } from './ReviewQueueList';
import type { ReviewQueueItem, ReviewStatus } from './types';

// The list looks up trace output previews to label rows; with no preview data it
// falls back to the target id, which keeps these assertions stable.
jest.mock('@databricks/web-shared/model-trace-explorer', () => ({
  useGetTracesById: () => ({ data: [] }),
}));

const renderWithProviders = (ui: React.ReactElement) =>
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>{ui}</DesignSystemProvider>
    </IntlProvider>,
  );

const item = (itemId: string, status: ReviewStatus, completedBy?: string): ReviewQueueItem => ({
  queue_id: 'rq-1',
  item_type: 'TRACE',
  item_id: itemId,
  status,
  completed_by: completedBy,
  creation_time_ms: 1_780_000_000_000,
  last_update_time_ms: 1_780_000_000_000,
});

const NOW = 1_780_000_010_000;

describe('ReviewQueueList', () => {
  it('renders pending traces in the To do group, hiding completed ones by default', () => {
    renderWithProviders(
      <ReviewQueueList
        items={[item('tr-1', 'PENDING'), item('tr-2', 'COMPLETE', 'bob')]}
        onOpen={jest.fn()}
        nowMs={NOW}
      />,
    );
    expect(screen.getByText('tr-1')).toBeInTheDocument();
    expect(screen.getByText('Needs review')).toBeInTheDocument();
    // Completed traces live in a collapsed "Completed" group.
    expect(screen.queryByText('tr-2')).not.toBeInTheDocument();
  });

  it('reveals completed traces when the Completed group is expanded', () => {
    renderWithProviders(<ReviewQueueList items={[item('tr-2', 'COMPLETE', 'bob')]} onOpen={jest.fn()} nowMs={NOW} />);
    expect(screen.queryByText('tr-2')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Completed'));
    expect(screen.getByText('tr-2')).toBeInTheDocument();
    expect(screen.getByText('Complete')).toBeInTheDocument();
    expect(screen.getByText('bob')).toBeInTheDocument();
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

  it('calls onOpen with the clicked trace', () => {
    const onOpen = jest.fn();
    renderWithProviders(<ReviewQueueList items={[item('tr-1', 'PENDING')]} onOpen={onOpen} nowMs={NOW} />);
    fireEvent.click(screen.getByText('tr-1'));
    expect(onOpen).toHaveBeenCalledTimes(1);
    expect(onOpen).toHaveBeenCalledWith(expect.objectContaining({ item_id: 'tr-1' }));
  });

  it('renders the declined status label once the Completed group is expanded', () => {
    renderWithProviders(<ReviewQueueList items={[item('tr-3', 'DECLINED', 'carol')]} onOpen={jest.fn()} nowMs={NOW} />);
    fireEvent.click(screen.getByText('Completed'));
    expect(screen.getByText('Declined')).toBeInTheDocument();
  });
});
