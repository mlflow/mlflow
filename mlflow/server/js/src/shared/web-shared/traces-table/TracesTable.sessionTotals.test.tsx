import { describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TracesTable, type TracesTableProps } from './TracesTable';
import { makeSessionTrace } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';
import { makeBaseTableProps } from './test-utils/tableProps';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// The standard fixture gives each turn a 1.2s duration and a $0.003 cost.
const props = (
  traces = [makeSessionTrace('s1-turn-1', 's1'), makeSessionTrace('s1-turn-2', 's1')],
): TracesTableProps => ({
  ...makeBaseTableProps(),
  traces,
  visibleColumns: ['duration', 'cost'],
  isGroupedBySession: true,
});

describe('TracesTable grouped session totals', () => {
  test('sums duration and cost in a grouped session header with the new eval runs UI', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await renderWithProviders(<TracesTable {...props()} />);

    expect(screen.getByText('2.4s')).toBeInTheDocument();
    expect(screen.getByText('$0.006')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: 'Expand session s1' }));
    expect(screen.getAllByText('1.2s')).toHaveLength(2);
  });

  test('leaves the session duration blank when a turn has no duration', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    const traces = [
      makeSessionTrace('s1-turn-1', 's1'),
      { ...makeSessionTrace('s1-turn-2', 's1'), execution_duration: undefined },
    ];
    await renderWithProviders(<TracesTable {...props(traces)} />);

    expect(screen.queryByText('1.2s')).not.toBeInTheDocument();
    expect(screen.getByText('$0.006')).toBeInTheDocument();
  });

  test('leaves the session duration and cost blank without the new eval runs UI', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    await renderWithProviders(<TracesTable {...props()} />);

    expect(screen.queryByText('2.4s')).not.toBeInTheDocument();
    expect(screen.queryByText('$0.006')).not.toBeInTheDocument();
  });
});
