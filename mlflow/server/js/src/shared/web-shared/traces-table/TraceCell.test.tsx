import { beforeEach, describe, expect, jest, test } from '@jest/globals';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { TraceIdCell } from './TraceCell';
import { makeTrace } from './test-utils/mockTraces';
import { renderWithProviders } from './test-utils/renderWithProviders';

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
