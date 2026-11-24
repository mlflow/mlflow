import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { TraceCostCell, TraceIdCell } from './TraceCell';
import { makeTrace } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';
import { shouldEnableNewEvalRunsUi } from '../../../common/utils/FeatureUtils';

jest.mock('../../../common/utils/FeatureUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/FeatureUtils')>('../../../common/utils/FeatureUtils'),
  shouldEnableNewEvalRunsUi: jest.fn(() => false),
}));

// Observe the clipboard write at its boundary: useCopyController copies via `use-clipboard-copy`'s
// `useClipboard().copy(text)`, and jsdom has no real clipboard. Stubbing this third-party library
// (not the internal copy hook/component, which run for real) lets us assert the exact text handed to
// the clipboard — the observable this cell's copyText change is about.
const mockClipboardCopy = jest.fn();
jest.mock('use-clipboard-copy', () => ({
  useClipboard: () => ({ copy: mockClipboardCopy }),
}));

describe('TraceIdCell copy button', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  // The copy button's copyText is `doesTraceSupportV4API(trace) ? createTraceV4LongIdentifier(trace) : trace_id`.
  // A trace with a UC location is V4-supported → the full `trace:/<location>/<id>`; one without a
  // trace_location is not → the bare id.
  test.each([
    {
      name: 'copies the full V4 identifier (with location prefix) for a V4-supported trace',
      trace: makeTrace('abc123'), // default fixture carries a UC_SCHEMA location (cat.sch)
      expectedCopy: 'trace:/cat.sch/abc123',
    },
    {
      name: 'copies the bare trace id for a non-V4 trace',
      trace: makeTrace('abc123', { trace_location: undefined }),
      expectedCopy: 'abc123',
    },
  ])('$name', async ({ trace, expectedCopy }) => {
    await renderWithProviders(
      <TraceIdCell trace={trace} onSelect={jest.fn()} accessibleLabel="Open trace abc123 — trace id" />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Copy' }));

    expect(mockClipboardCopy).toHaveBeenCalledWith(expectedCopy);
  });
});

describe('TraceCostCell', () => {
  test('flag on: renders the ≈-prefixed cost with an accessible label and an estimation tooltip', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(true);
    await renderWithProviders(<TraceCostCell trace={makeTrace('cost-trace')} />);

    expect(screen.getByText('≈$0.003')).toBeVisible();
    expect(screen.getByLabelText('Estimated cost: $0.003')).toBeVisible();
    await userEvent.hover(screen.getByText('≈$0.003'));
    const tooltip = await screen.findByRole('tooltip');
    expect(tooltip).toHaveTextContent('Input $0.001 · Output $0.002');
    expect(tooltip).toHaveTextContent('Cost is estimated from token usage and public model pricing.');
  });

  test('flag off: renders the plain cost', async () => {
    jest.mocked(shouldEnableNewEvalRunsUi).mockReturnValue(false);
    await renderWithProviders(<TraceCostCell trace={makeTrace('cost-trace')} />);

    expect(screen.getByText('$0.003')).toBeVisible();
    expect(screen.queryByText('≈$0.003')).not.toBeInTheDocument();
  });
});
