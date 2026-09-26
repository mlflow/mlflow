import { jest, describe, beforeEach, test, expect } from '@jest/globals';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';
import ShowArtifactMarkdownView from './ShowArtifactMarkdownView';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

jest.mock('./utils/fetchArtifactUnified', () => ({
  fetchArtifactUnified: jest.fn(),
}));

const mockFetch = jest.mocked(fetchArtifactUnified);
const mockMermaidInitialize = jest.fn();
const mockMermaidRender = jest.fn<(...args: string[]) => Promise<{ svg: string }>>();

jest.mock('mermaid', () => ({
  __esModule: true,
  default: {
    initialize: mockMermaidInitialize,
    render: mockMermaidRender,
  },
}));

const renderView = (props = {}) =>
  renderWithDesignSystem(<ShowArtifactMarkdownView runUuid="run-1" path="notes.md" {...props} />);

describe('ShowArtifactMarkdownView', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockMermaidRender.mockResolvedValue({ svg: '<svg role="img"><text>Diagram</text></svg>' });
  });

  test('shows skeleton while loading', () => {
    mockFetch.mockReturnValue(new Promise(() => {}));
    renderView();
    expect(document.querySelector('.artifact-markdown-view-loading')).toBeInTheDocument();
  });

  test('renders markdown as formatted content', async () => {
    mockFetch.mockResolvedValue('# Title\n\nSome **bold** text.');
    renderView();
    await waitFor(() => {
      expect(screen.getByText('Title')).toBeInTheDocument();
    });
    expect(screen.getByText(/bold/)).toBeInTheDocument();
  });

  test('renders tables via remark-gfm', async () => {
    mockFetch.mockResolvedValue('| A | B |\n|---|---|\n| 1 | 2 |');
    renderView();
    await waitFor(() => {
      expect(screen.getByText('A')).toBeInTheDocument();
    });
    expect(screen.getByText('1')).toBeInTheDocument();
  });

  test('shows error state on fetch failure', async () => {
    mockFetch.mockRejectedValue(new Error('network error'));
    renderView();
    await waitFor(() => {
      expect(document.querySelector('.artifact-markdown-view-error')).toBeInTheDocument();
    });
  });

  test('refetches when path changes', async () => {
    mockFetch.mockResolvedValue('first');
    const { rerender } = renderView({ path: 'a.md' });
    await waitFor(() => expect(screen.getByText('first')).toBeInTheDocument());

    mockFetch.mockResolvedValue('second');
    rerender(<ShowArtifactMarkdownView runUuid="run-1" path="b.md" />);
    await waitFor(() => expect(screen.getByText('second')).toBeInTheDocument());
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  test('toggles between rendered and source views', async () => {
    const rawMd = '# Title\n\nSome **bold** text.';
    mockFetch.mockResolvedValue(rawMd);
    renderView();

    await waitFor(() => {
      expect(screen.getByText('Title')).toBeInTheDocument();
    });
    // Rendered heading should be in an h1, not a <pre>
    expect(screen.getByText('Title').tagName).toBe('H1');

    // Switch to source view
    await userEvent.click(screen.getByTestId('markdown-view-source-button'));

    // Raw markdown source should be visible in a <pre>
    const pre = document.querySelector('pre');
    expect(pre).toBeInTheDocument();
    expect(pre?.textContent).toBe(rawMd);

    // Switch back to rendered view
    await userEvent.click(screen.getByTestId('markdown-view-rendered-button'));
    await waitFor(() => {
      expect(screen.getByText('Title').tagName).toBe('H1');
    });
  });

  test('renders Mermaid fenced blocks with strict security settings', async () => {
    mockFetch.mockResolvedValue('```mermaid\ngraph TD\n  A --> B\n```');
    renderView();

    await waitFor(() => expect(screen.getByTestId('mermaid-diagram')).toBeInTheDocument());

    expect(mockMermaidInitialize).toHaveBeenCalledWith(
      expect.objectContaining({
        securityLevel: 'strict',
        startOnLoad: false,
        htmlLabels: false,
        theme: 'default',
        secure: expect.arrayContaining(['securityLevel', 'theme', 'themeCSS', 'themeVariables']),
      }),
    );
    expect(mockMermaidRender).toHaveBeenCalledWith(
      expect.stringMatching(/^mlflow-mermaid-\d+$/),
      'graph TD\n  A --> B',
    );
    expect(screen.getByText('Diagram')).toBeInTheDocument();
  });

  test('leaves ordinary fenced code blocks unchanged', async () => {
    mockFetch.mockResolvedValue('```python\nprint("hello")\n```');
    renderView();

    await waitFor(() => expect(document.querySelector('code')?.textContent).toContain('print("hello")'));

    expect(mockMermaidRender).not.toHaveBeenCalled();
  });

  test('does not load Mermaid while a large artifact remains in source mode', async () => {
    const rawMd = '```mermaid\ngraph TD\n  A --> B\n```';
    mockFetch.mockResolvedValue(rawMd);
    renderView({ size: 101 * 1024 });

    await waitFor(() => expect(document.querySelector('pre')?.textContent).toContain(rawMd));
    expect(mockMermaidRender).not.toHaveBeenCalled();

    await userEvent.click(screen.getByTestId('markdown-view-rendered-button'));
    await waitFor(() => expect(mockMermaidRender).toHaveBeenCalledTimes(1));
  });

  test('falls back to the Mermaid source when rendering fails', async () => {
    mockFetch.mockResolvedValue('```mermaid\ninvalid diagram\n```');
    mockMermaidRender.mockRejectedValue(new Error('parse error'));
    renderView();

    await waitFor(() =>
      expect(screen.getByText('Unable to render Mermaid diagram. Showing source instead.')).toBeInTheDocument(),
    );
    expect(screen.getByText('invalid diagram')).toBeInTheDocument();
  });

  test('sanitizes rendered Mermaid SVG', async () => {
    mockFetch.mockResolvedValue('```mermaid\ngraph TD\n  A\n```');
    mockMermaidRender.mockResolvedValue({
      svg: '<svg role="img" onload="alert(1)"><script>alert(1)</script><text>Safe</text></svg>',
    });
    renderView();

    await waitFor(() => expect(screen.getByTestId('mermaid-diagram')).toBeInTheDocument());

    const diagram = screen.getByTestId('mermaid-diagram');
    expect(diagram.querySelector('script')).not.toBeInTheDocument();
    expect(diagram.querySelector('svg')).not.toHaveAttribute('onload');
    expect(screen.getByText('Safe')).toBeInTheDocument();
  });

  test('uses a unique ID for each Mermaid diagram', async () => {
    mockFetch.mockResolvedValue('```mermaid\ngraph TD\n  A\n```\n\n```mermaid\ngraph TD\n  B\n```');
    renderView();

    await waitFor(() => expect(mockMermaidRender).toHaveBeenCalledTimes(2));

    const firstId = mockMermaidRender.mock.calls[0][0];
    const secondId = mockMermaidRender.mock.calls[1][0];
    expect(firstId).not.toBe(secondId);
  });
});
