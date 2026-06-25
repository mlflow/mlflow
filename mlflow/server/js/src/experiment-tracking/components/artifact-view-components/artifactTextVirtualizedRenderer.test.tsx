import { describe, jest, test, expect } from '@jest/globals';
import React from 'react';
import { render, screen } from '../../../common/utils/TestUtils.react18';
import {
  virtualizedRenderer,
  ARTIFACT_TEXT_VIRTUALIZED_SPACER_TESTID,
  ARTIFACT_TEXT_VIRTUALIZED_ROW_TESTID,
} from './artifactTextVirtualizedRenderer';

jest.mock('@tanstack/react-virtual', () => {
  const actual = jest.requireActual<typeof import('@tanstack/react-virtual')>('@tanstack/react-virtual');
  return {
    ...actual,
    useVirtualizer: (opts: any) => ({
      getVirtualItems: () =>
        Array.from({ length: Math.min(opts.count, 50) }, (_, i) => ({
          index: i,
          key: i,
          start: i * 20,
          size: 20,
        })),
      getTotalSize: () => opts.count * 20,
      measureElement: () => {},
    }),
  };
});

describe('artifactTextVirtualizedRenderer', () => {
  const mockStylesheet = {};

  const makeRows = (count: number) =>
    Array.from({ length: count }, (_, i) => ({
      type: 'element',
      tagName: 'span',
      properties: { className: [] },
      children: [{ type: 'text', value: `line ${i}` }],
    }));

  test('renders visible rows from the rows array', () => {
    const rows = makeRows(3);

    const { container } = render(
      <div>{virtualizedRenderer({ rows, stylesheet: mockStylesheet, useInlineStyles: true })}</div>,
    );

    expect(screen.getByText('line 0')).toBeInTheDocument();
    expect(screen.getByText('line 1')).toBeInTheDocument();
    expect(screen.getByText('line 2')).toBeInTheDocument();
    expect(container.querySelectorAll(`[data-testid="${ARTIFACT_TEXT_VIRTUALIZED_ROW_TESTID}"]`)).toHaveLength(3);
  });

  test('renders no rows when rows array is empty', () => {
    const { container } = render(
      <div>{virtualizedRenderer({ rows: [], stylesheet: mockStylesheet, useInlineStyles: true })}</div>,
    );

    expect(container.querySelectorAll(`[data-testid="${ARTIFACT_TEXT_VIRTUALIZED_ROW_TESTID}"]`)).toHaveLength(0);
  });

  test('virtualizes large row sets - spacer reflects total height while rendering subset', () => {
    const rows = makeRows(200);

    const { container } = render(
      <div>{virtualizedRenderer({ rows, stylesheet: mockStylesheet, useInlineStyles: true })}</div>,
    );

    // The spacer height should reflect all 200 rows (200 * 20 = 4000px).
    const spacer = container.querySelector<HTMLElement>(`[data-testid="${ARTIFACT_TEXT_VIRTUALIZED_SPACER_TESTID}"]`);
    expect(spacer).not.toBeNull();
    expect(spacer?.style.height).toBe('4000px');
    // Only the virtualized subset is mounted in the DOM. The mock caps getVirtualItems() at 50.
    const renderedRows = container.querySelectorAll(`[data-testid="${ARTIFACT_TEXT_VIRTUALIZED_ROW_TESTID}"]`);
    expect(renderedRows).toHaveLength(50);
  });
});
