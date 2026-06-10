import { describe, jest, test, expect } from '@jest/globals';
import React from 'react';
import { render, screen } from '../../../common/utils/TestUtils.react18';
import { virtualizedRenderer } from './artifactTextVirtualizedRenderer';

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
    }),
  };
});

describe('artifactTextVirtualizedRenderer', () => {
  const mockStylesheet = {};

  test('virtualizedRenderer returns a function', () => {
    const renderer = virtualizedRenderer();
    expect(typeof renderer).toBe('function');
  });

  test('renders visible rows from the rows array', () => {
    const rows = [
      {
        type: 'element',
        tagName: 'span',
        properties: { className: [] },
        children: [{ type: 'text', value: 'line one' }],
      },
      {
        type: 'element',
        tagName: 'span',
        properties: { className: [] },
        children: [{ type: 'text', value: 'line two' }],
      },
      {
        type: 'element',
        tagName: 'span',
        properties: { className: [] },
        children: [{ type: 'text', value: 'line three' }],
      },
    ];

    const renderer = virtualizedRenderer();
    const { container } = render(<div>{renderer({ rows, stylesheet: mockStylesheet, useInlineStyles: true })}</div>);

    expect(screen.getByText('line one')).toBeInTheDocument();
    expect(screen.getByText('line two')).toBeInTheDocument();
    expect(screen.getByText('line three')).toBeInTheDocument();
    expect(container.querySelector('[style*="translateY"]')).toBeInTheDocument();
  });

  test('renders empty when rows array is empty', () => {
    const renderer = virtualizedRenderer();
    const { container } = render(
      <div>{renderer({ rows: [], stylesheet: mockStylesheet, useInlineStyles: true })}</div>,
    );

    // Should have the container divs but no row content
    expect(container.querySelectorAll('[style*="translateY"]')).toHaveLength(0);
  });

  test('virtualizes large row sets - container reflects total height while rendering subset', () => {
    const rows = Array.from({ length: 200 }, (_, i) => ({
      type: 'element',
      tagName: 'span',
      properties: { className: [] },
      children: [{ type: 'text', value: `line ${i}` }],
    }));

    const renderer = virtualizedRenderer();
    const { container } = render(<div>{renderer({ rows, stylesheet: mockStylesheet, useInlineStyles: true })}</div>);

    // Total container height should reflect all 200 rows (200 * 20 = 4000px)
    expect(container.querySelector('[style*="height: 4000px"]')).toBeInTheDocument();
    // Only the virtualized subset is rendered in the DOM. The mock caps getVirtualItems() at 50.
    const renderedRows = container.querySelectorAll('[style*="translateY"]');
    expect(renderedRows.length).toBe(50);
  });
});
