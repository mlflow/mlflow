import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithDesignSystem } from '../../../common/utils/TestUtils.react18';
import ShowArtifactMarkdownView from './ShowArtifactMarkdownView';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

jest.mock('./utils/fetchArtifactUnified', () => ({
  fetchArtifactUnified: jest.fn(),
}));

const mockFetch = fetchArtifactUnified as jest.MockedFunction<typeof fetchArtifactUnified>;

const renderView = (props = {}) =>
  renderWithDesignSystem(<ShowArtifactMarkdownView runUuid="run-1" path="notes.md" experimentId="0" {...props} />);

describe('ShowArtifactMarkdownView', () => {
  beforeEach(() => jest.clearAllMocks());

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
    rerender(<ShowArtifactMarkdownView runUuid="run-1" path="b.md" experimentId="0" />);
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
});
