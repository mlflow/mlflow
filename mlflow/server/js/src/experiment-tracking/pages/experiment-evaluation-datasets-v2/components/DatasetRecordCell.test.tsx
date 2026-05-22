/* eslint-disable @typescript-eslint/ban-ts-comment */
// @ts-nocheck — punting test typing; see PR2 plan in branch import { describe, expect, jest, test } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import type { ReactNode } from 'react';
import { JsonPreviewCell, TagsPreviewCell, getTruncationComment, truncateJsonForTooltip } from './DatasetRecordCell';
import { jest } from '@jest/globals';
import { describe } from '@jest/globals';
import { test } from '@jest/globals';
import { expect } from '@jest/globals';

const wrap = ({ children }: { children: ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const renderTagsPreview = (props: Partial<React.ComponentProps<typeof TagsPreviewCell>> = {}) => {
  const onActivate = props.onActivate ?? jest.fn();
  const accessibleLabel = props.accessibleLabel ?? 'Open dataset record rec-1 — tags';
  return {
    onActivate,
    ...render(<TagsPreviewCell tags={props.tags} onActivate={onActivate} accessibleLabel={accessibleLabel} />, {
      wrapper: wrap,
    }),
  };
};

describe('TagsPreviewCell', () => {
  test('renders a "-" placeholder and no activator when there are no tags', () => {
    renderTagsPreview({ tags: undefined });
    expect(screen.getByText('-')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('renders a single pill with key: value when there is one tag — no "+N more", no tooltip', () => {
    renderTagsPreview({ tags: { env: 'prod' } });
    expect(screen.getByText('env: prod')).toBeInTheDocument();
    expect(screen.queryByText(/more$/i)).not.toBeInTheDocument();
  });

  test('renders first tag plus "+N more" suffix when there are multiple tags', () => {
    renderTagsPreview({ tags: { env: 'prod', region: 'us-west-2', team: 'ml' } });
    expect(screen.getByText('env: prod')).toBeInTheDocument();
    expect(screen.getByText('+2 more')).toBeInTheDocument();
    // The hidden tags are not displayed inline — they live in the hover tooltip.
    expect(screen.queryByText('region: us-west-2')).not.toBeInTheDocument();
  });

  test('clicking the activator fires onActivate', async () => {
    const user = userEvent.setup();
    const { onActivate } = renderTagsPreview({ tags: { env: 'prod' } });
    await user.click(screen.getByRole('button', { name: /Open dataset record rec-1 — tags/ }));
    expect(onActivate).toHaveBeenCalledTimes(1);
  });

  test('Enter and Space on the activator fire onActivate', async () => {
    const user = userEvent.setup();
    const { onActivate } = renderTagsPreview({ tags: { env: 'prod' } });
    const activator = screen.getByRole('button', { name: /Open dataset record rec-1 — tags/ });
    activator.focus();
    await user.keyboard('{Enter}');
    await user.keyboard(' ');
    expect(onActivate).toHaveBeenCalledTimes(2);
  });

  test('aria-label propagates to the activator', () => {
    renderTagsPreview({ tags: { env: 'prod' }, accessibleLabel: 'Open dataset record rec-42 — tags' });
    expect(screen.getByRole('button', { name: 'Open dataset record rec-42 — tags' })).toBeInTheDocument();
  });
});

describe('JsonPreviewCell', () => {
  test('renders the supplied emptyLabel when value is empty', () => {
    render(<JsonPreviewCell value={undefined} emptyLabel="(empty)" onActivate={jest.fn()} accessibleLabel="x" />, {
      wrapper: wrap,
    });
    expect(screen.getByText('(empty)')).toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('renders compact JSON in a monospace <span> — not a <code> element', () => {
    render(
      <JsonPreviewCell
        value={{ question: 'hi' }}
        emptyLabel="(empty)"
        onActivate={jest.fn()}
        accessibleLabel="Open dataset record rec-1 — inputs"
      />,
      { wrapper: wrap },
    );
    const text = screen.getByText('{"question":"hi"}');
    // Locks in the requirement that inputs/expectations are rendered as monospace text,
    // not in code blocks. If someone reverts to <code>, this test catches it.
    expect(text.tagName.toLowerCase()).toBe('span');
  });

  test('clicking the activator fires onActivate', async () => {
    const user = userEvent.setup();
    const onActivate = jest.fn();
    render(
      <JsonPreviewCell
        value={{ question: 'hi' }}
        emptyLabel="(empty)"
        onActivate={onActivate}
        accessibleLabel="Open dataset record rec-1 — inputs"
      />,
      { wrapper: wrap },
    );
    await user.click(screen.getByRole('button', { name: /Open dataset record rec-1 — inputs/ }));
    expect(onActivate).toHaveBeenCalledTimes(1);
  });
});

// Tooltip body content is driven by `truncateJsonForTooltip` and rendered through a
// recursive node renderer. Radix Tooltip's portal doesn't reliably open in jsdom on
// `user.hover` (the visible tooltip never mounts), so we cover the truncation
// algorithm directly. The rendered styled tree is covered by the manual browser
// smoke test in the implementation plan.
describe('truncateJsonForTooltip', () => {
  test('clips long string values to 30 chars + ellipsis', () => {
    const truncated = truncateJsonForTooltip({ field: 'x'.repeat(200) }) as { field: string };
    expect(truncated.field).toBe('x'.repeat(30) + '…');
  });

  test('passes short string values through untouched', () => {
    const truncated = truncateJsonForTooltip({ field: 'short' }) as { field: string };
    expect(truncated.field).toBe('short');
  });

  test('caps arrays at 2 items and exposes a "+N more" trailing comment via getTruncationComment', () => {
    const truncated = truncateJsonForTooltip({ items: [0, 1, 2, 3, 4] }) as { items: unknown[] };
    // The marker is stored out-of-band (WeakMap) so it doesn't appear as an element —
    // the renderer reads it separately and emits a comment-style line. This guards against
    // regressing to the in-band `[..., '…+N more']` shape which made the marker look like a
    // real string element when rendered.
    expect(truncated.items).toEqual([0, 1]);
    expect(getTruncationComment(truncated.items)).toBe('+3 more');
  });

  test('leaves arrays shorter than the cap untouched, with no truncation marker', () => {
    const truncated = truncateJsonForTooltip({ items: [0, 1] }) as { items: unknown[] };
    expect(truncated.items).toEqual([0, 1]);
    expect(getTruncationComment(truncated.items)).toBeUndefined();
  });

  test('caps object keys at 2 and exposes a "+N more keys" trailing comment via getTruncationComment', () => {
    const truncated = truncateJsonForTooltip({ a: 1, b: 2, c: 3, d: 4, e: 5 }) as Record<string, unknown>;
    // Real entries only — the marker lives out-of-band (WeakMap) and never shows up as a
    // string key. This guards against regressing to the in-band `'…': '+N more keys'`
    // shape which rendered as a fake `"…": "+2 more keys"` entry.
    expect(Object.keys(truncated)).toEqual(['a', 'b']);
    expect(truncated['a']).toBe(1);
    expect(truncated['b']).toBe(2);
    expect(getTruncationComment(truncated)).toBe('+3 more keys');
  });

  test('preserves primitive scalars (number, boolean, null) verbatim when under the key cap', () => {
    // Two keys keep us under TOOLTIP_MAX_OBJECT_KEYS so we can validate scalar pass-through
    // without also tripping the object-key truncation path.
    expect(truncateJsonForTooltip({ n: 42, b: true })).toEqual({ n: 42, b: true });
    expect(truncateJsonForTooltip({ z: null, n: 0 })).toEqual({ z: null, n: 0 });
  });

  test('result is always valid JSON for a deeply mixed input', () => {
    const value = {
      summary: 'lorem ipsum '.repeat(20),
      tags: ['a', 'b', 'c', 'd'],
      meta: { k1: 'v1', k2: 'v2', k3: 'v3' },
    };
    const json = JSON.stringify(truncateJsonForTooltip(value));
    expect(() => JSON.parse(json)).not.toThrow();
  });
});
