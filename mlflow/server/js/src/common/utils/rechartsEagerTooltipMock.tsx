import React from 'react';

/**
 * Single payload entry the eager Tooltip mock will inject. Mirrors the shape
 * Recharts produces at runtime, but only includes the fields downstream tooltip
 * content components actually read.
 */
export interface EagerTooltipPayloadEntry {
  payload: Record<string, unknown>;
  name?: string;
  value?: number | string;
  color?: string;
}

/**
 * Builds a Recharts mock module identical to the global auto-mock at
 * `mlflow/server/js/__mocks__/recharts.tsx`, except that <Tooltip> eagerly
 * renders its `content` prop with synthetic `active=true` and the supplied
 * payload.
 *
 * Use when a chart-level test needs to interact with content rendered inside
 * <Tooltip> (e.g. clicking the "View traces for this period" link inside
 * ScrollableTooltip and asserting on the resulting navigation URL). The
 * default mock stubs <Tooltip> as `<div data-testid="tooltip" />`, which makes
 * such interactions impossible.
 *
 * Designed to be returned from a `jest.mock('recharts', () => ...)` factory.
 * The factory must reach this helper via `jest.requireActual`, because
 * `jest.mock` calls are hoisted above regular imports:
 *
 *   jest.mock('recharts', () => {
 *     const { buildEagerTooltipRechartsMock } = jest.requireActual<
 *       typeof import('<relative-path>/common/utils/rechartsEagerTooltipMock')
 *     >('<relative-path>/common/utils/rechartsEagerTooltipMock');
 *     return buildEagerTooltipRechartsMock([
 *       { payload: { timestampMs: ... }, name: 'count', value: 42, color: 'blue' },
 *     ]);
 *   });
 *
 * The returned module spreads the global auto-mock as its base, so any new
 * component added to `__mocks__/recharts.tsx` is automatically picked up by
 * consumers of this helper without needing to update each test file.
 */
export function buildEagerTooltipRechartsMock(payload: EagerTooltipPayloadEntry[]) {
  const baseMock = jest.requireActual<typeof import('../../../__mocks__/recharts')>('../../../__mocks__/recharts');
  return {
    ...baseMock,
    Tooltip: ({ content }: { content?: React.ReactElement }) =>
      content ? React.cloneElement(content, { active: true, payload }) : <div data-testid="tooltip" />,
  };
}
