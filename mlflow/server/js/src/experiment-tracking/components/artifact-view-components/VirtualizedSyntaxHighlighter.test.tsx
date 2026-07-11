import { describe, beforeEach, jest, test, expect } from '@jest/globals';
import React from 'react';
import { render, screen } from '@testing-library/react';
import { coy } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import { VirtualizedSyntaxHighlighter } from './VirtualizedSyntaxHighlighter';

const VISIBLE_LINE_INDICES = [0, 1, 2];
const VIRTUAL_ITEM_SIZE = 20;

jest.mock('@tanstack/react-virtual', () => {
  const actual = jest.requireActual<typeof import('@tanstack/react-virtual')>('@tanstack/react-virtual');
  return {
    ...actual,
    useVirtualizer: (opts: any) => ({
      getVirtualItems: () =>
        VISIBLE_LINE_INDICES.map((index) => ({
          index,
          key: `line-${index}`,
          start: index * VIRTUAL_ITEM_SIZE,
          size: VIRTUAL_ITEM_SIZE,
          measureElement: () => {},
        })),
      getTotalSize: () => opts.count * VIRTUAL_ITEM_SIZE,
      measureElement: () => {},
    }),
  };
});

describe('VirtualizedSyntaxHighlighter', () => {
  const renderComponent = (lineCount: number) => {
    const content = Array.from({ length: lineCount }, (_, i) => `line ${i}`).join('\n');
    return render(
      <div style={{ width: 400, height: 200 }}>
        <VirtualizedSyntaxHighlighter language="text" style={coy}>
          {content}
        </VirtualizedSyntaxHighlighter>
      </div>,
    );
  };

  test('renders only the virtual items returned by the virtualizer', () => {
    renderComponent(100);

    expect(screen.getByText('line 0')).toBeInTheDocument();
    expect(screen.getByText('line 1')).toBeInTheDocument();
    expect(screen.getByText('line 2')).toBeInTheDocument();
    expect(screen.queryByText('line 3')).not.toBeInTheDocument();
    expect(screen.queryByText('line 99')).not.toBeInTheDocument();
  });

  test('sets the spacer height to the total virtual size', () => {
    renderComponent(100);
    const spacer = screen.getByTestId('virtualized-syntax-highlighter-spacer');
    expect(spacer).toBeInTheDocument();
    expect(spacer.style.height).toBe(`${100 * VIRTUAL_ITEM_SIZE}px`);
  });
});
