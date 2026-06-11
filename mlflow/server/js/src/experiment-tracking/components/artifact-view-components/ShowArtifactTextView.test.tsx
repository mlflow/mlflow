import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';
import ShowArtifactTextView, { isLineHeavy } from './ShowArtifactTextView';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

jest.mock('./utils/fetchArtifactUnified', () => ({
  fetchArtifactUnified: jest.fn(),
}));

// Render a fixed window of rows in tests; jsdom has no layout so the real
// virtualizer would measure a zero-height scroll element and render nothing.
const VIRTUALIZER_TEST_WINDOW = 50;
jest.mock('@tanstack/react-virtual', () => {
  const actual = jest.requireActual<typeof import('@tanstack/react-virtual')>('@tanstack/react-virtual');
  return {
    ...actual,
    useVirtualizer: (opts: any) => ({
      getVirtualItems: () =>
        Array.from({ length: Math.min(opts.count, VIRTUALIZER_TEST_WINDOW) }, (_, i) => ({
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

const mockFetch = jest.mocked(fetchArtifactUnified);

const renderView = (props = {}) =>
  renderWithDesignSystem(<ShowArtifactTextView runUuid="run-1" path="output.log" {...props} />);

describe('ShowArtifactTextView virtualization', () => {
  beforeEach(() => jest.clearAllMocks());

  test('renders small files fully without virtualization', async () => {
    mockFetch.mockResolvedValue('line one\nline two\nline three');
    renderView();
    await waitFor(() => {
      expect(screen.getByText(/line one/)).toBeInTheDocument();
    });
    expect(screen.getByText(/line three/)).toBeInTheDocument();
    // Non-virtualized path renders no windowed row containers
    expect(document.querySelectorAll('[data-index]')).toHaveLength(0);
  });

  test('renders a line-heavy file without RangeError and mounts only the visible window', async () => {
    // 140K lines exceeds V8's ~125K argument limit that crashes
    // react-syntax-highlighter's default renderer
    const lineCount = 140_000;
    const text = Array.from({ length: lineCount }, (_, i) => `line-${i}`).join('\n');
    mockFetch.mockResolvedValue(text);

    renderView({ size: text.length });

    await waitFor(() => {
      expect(screen.getByText(/line-0/)).toBeInTheDocument();
    });
    const mountedRows = document.querySelectorAll('[data-index]');
    expect(mountedRows.length).toBeGreaterThan(0);
    expect(mountedRows.length).toBeLessThanOrEqual(VIRTUALIZER_TEST_WINDOW);
  });

  test('virtualized path preserves the pre/code structure and syntax highlighting', async () => {
    const text = Array.from({ length: 6000 }, (_, i) => `row ${i}`).join('\n');
    mockFetch.mockResolvedValue(text);
    renderView({ size: text.length });

    await waitFor(() => {
      expect(document.querySelector('pre code')).toBeInTheDocument();
    });
    // Line content is preserved (tokenized into multiple spans by the `log` grammar,
    // so compare textContent rather than looking up a single text node)
    expect(document.querySelector('[data-index="0"]')?.textContent).toBe('row 0\n');
    expect(document.querySelector('[data-index="1"]')?.textContent).toBe('row 1\n');
  });
});

describe('isLineHeavy', () => {
  test('returns false for files at or below the threshold', () => {
    expect(isLineHeavy('')).toBe(false);
    expect(isLineHeavy('one line')).toBe(false);
    expect(isLineHeavy(Array(5000).fill('x').join('\n'))).toBe(false);
  });

  test('returns true for files above the threshold', () => {
    expect(isLineHeavy(Array(5001).fill('x').join('\n'))).toBe(true);
  });
});
