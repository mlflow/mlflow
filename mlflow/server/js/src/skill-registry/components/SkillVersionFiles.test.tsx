import { beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { rest } from 'msw';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';

import { setupServer } from '../../common/utils/setup-msw';
import { createMockSkillVersion } from '../test-utils';
import { SkillManifestPreview, SkillVersionFiles } from './SkillVersionFiles';

const ROOT = 'skills/@acme/code-review/0123456789abcdef0123456789abcdef';
const OTHER_ROOT = 'skills/@acme/code-review/fedcba9876543210fedcba9876543210';
const LISTINGS: Partial<Record<string, { path: string; is_dir?: boolean; file_size?: number }[]>> = {
  [ROOT]: [
    { path: 'scripts', is_dir: true },
    { path: 'README.md', is_dir: false, file_size: 40 },
    { path: 'SKILL.md', is_dir: false, file_size: 2048 },
    { path: 'huge.bin', is_dir: false, file_size: 10 * 1024 * 1024 },
  ],
  [`${ROOT}/scripts`]: [{ path: 'run.py', is_dir: false, file_size: 12 }],
  [OTHER_ROOT]: [{ path: 'scripts', is_dir: true }],
  // The same file path, now too large to preview.
  [`${OTHER_ROOT}/scripts`]: [{ path: 'run.py', is_dir: false, file_size: 10 * 1024 * 1024 }],
};
const CONTENTS: Partial<Record<string, string>> = {
  [`${ROOT}/SKILL.md`]:
    '---\nname: code-review\ndescription: Reviews a diff for bugs.\n---\n# Code review\n\nSee the [guide](https://example.com/guide).\n',
  [`${ROOT}/scripts/run.py`]: 'print("hi")\n',
  [`${ROOT}/README.md`]: 'See the ![diagram](https://example.com/diagram.png).\n',
};

// jsdom's fetch has no streaming body, so serve file content directly as other artifact tests do.
const mockGetArtifactChunkedText = jest.fn<(url: string) => Promise<string>>();
jest.mock('../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../common/utils/ArtifactUtils')>('../../common/utils/ArtifactUtils'),
  getArtifactChunkedText: (url: string) => mockGetArtifactChunkedText(url),
}));

const uploaded = createMockSkillVersion({
  source_type: 'mlflow',
  source: `mlflow-artifacts:/${ROOT}`,
  ref: null,
  subpath: null,
});

describe('SkillVersionFiles', () => {
  beforeEach(() => {
    mockGetArtifactChunkedText.mockImplementation(async (url) => {
      const content = CONTENTS[decodeURIComponent(url.split('/mlflow-artifacts/artifacts/')[1])];
      if (content === undefined) throw new Error('missing');
      return content;
    });
  });

  setupServer(
    rest.get(/mlflow-artifacts\/artifacts$/, (req, res, ctx) =>
      res(ctx.json({ files: LISTINGS[req.url.searchParams.get('path') ?? ''] ?? [] })),
    ),
  );

  const renderFiles = (version = uploaded) => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const ui = (shown: typeof version) => (
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={client}>
            <SkillVersionFiles version={shown} />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>
    );
    const result = render(ui(version));
    return { ...result, showVersion: (next: typeof version) => result.rerender(ui(next)) };
  };

  it('lists stored files with SKILL.md first and folders after files', async () => {
    renderFiles();

    await screen.findByText('SKILL.md');
    const rows = screen.getAllByRole('button').map((row) => row.textContent);
    expect(rows).toEqual(['SKILL.md2.0 KB', 'huge.bin10.0 MB', 'README.md40 B', 'scripts', 'run.py12 B']);

    await userEvent.click(screen.getByRole('button', { name: 'scripts' }));
    expect(screen.queryByText('run.py')).not.toBeInTheDocument();
  });

  it('shows a markdown file raw first and renders it on request, frontmatter as a table', async () => {
    renderFiles();

    await userEvent.click(await screen.findByText('SKILL.md'));
    const dialog = await screen.findByRole('dialog', { name: 'SKILL.md' });
    expect(await within(dialog).findByText(/# Code review/)).toBeInTheDocument();
    expect(within(dialog).getByRole('button', { name: 'Copy file contents' })).toBeInTheDocument();

    await userEvent.click(within(dialog).getByText('Preview'));
    expect(await within(dialog).findByRole('heading', { name: 'Code review' })).toBeInTheDocument();
    expect(within(dialog).getByText('name')).toBeInTheDocument();
    expect(within(dialog).getByText('code-review')).toBeInTheDocument();
    expect(within(dialog).queryByRole('button', { name: 'Copy file contents' })).not.toBeInTheDocument();
  });

  it('wraps code blocks in a rendered markdown file instead of scrolling them, each with a copy button', async () => {
    mockGetArtifactChunkedText.mockResolvedValueOnce('# Usage\n\n```py\nprint("hi")\n```\n');
    renderFiles();

    await userEvent.click(await screen.findByText('README.md'));
    const dialog = await screen.findByRole('dialog', { name: 'README.md' });
    await userEvent.click(await within(dialog).findByText('Preview'));
    const copy = await within(dialog).findByRole('button', { name: 'Copy code' });
    expect(dialog.querySelectorAll('pre')).toHaveLength(1);
    const block = copy.parentElement?.querySelector('pre');
    expect(block).toHaveTextContent('print("hi")');
    expect(block?.style.overflow).toBe('visible');
    expect(block?.querySelector('code')?.style.whiteSpace).toBe('pre-wrap');
  });

  it('shows frontmatter that uses YAML aliases as text instead of expanding it', async () => {
    mockGetArtifactChunkedText.mockResolvedValueOnce('---\nmeta: &m {self: *m}\n---\n# Doc\n');
    renderFiles();

    await userEvent.click(await screen.findByText('README.md'));
    const dialog = await screen.findByRole('dialog', { name: 'README.md' });
    await userEvent.click(await within(dialog).findByText('Preview'));
    expect(await within(dialog).findByRole('heading', { name: 'Doc' })).toBeInTheDocument();
    expect(within(dialog).getByText('meta: &m {self: *m}')).toBeInTheDocument();
  });

  it('shows images in markdown as links instead of loading them', async () => {
    renderFiles();

    await userEvent.click(await screen.findByText('README.md'));
    const dialog = await screen.findByRole('dialog', { name: 'README.md' });
    await userEvent.click(await within(dialog).findByText('Preview'));
    expect(await within(dialog).findByRole('link', { name: 'diagram' })).toHaveAttribute(
      'href',
      'https://example.com/diagram.png',
    );
    expect(dialog.querySelector('img')).toBeNull();
  });

  it('shows other files raw, without the view switch', async () => {
    renderFiles();

    await userEvent.click(await screen.findByText('run.py'));
    const dialog = await screen.findByRole('dialog', { name: 'scripts/run.py' });
    expect(await within(dialog).findByRole('button', { name: 'Copy file contents' })).toBeInTheDocument();
    expect(within(dialog).queryByText('Preview')).not.toBeInTheDocument();
  });

  it('previews a stored file with line numbers', async () => {
    renderFiles();

    await userEvent.click(await screen.findByText('run.py'));
    const dialog = await screen.findByRole('dialog', { name: 'scripts/run.py' });
    await waitFor(() => expect(within(dialog).getByText(/print/)).toBeInTheDocument());
  });

  it('closes an open preview when another version is shown', async () => {
    const { showVersion } = renderFiles();

    await userEvent.click(await screen.findByText('run.py'));
    expect(await screen.findByRole('dialog', { name: 'scripts/run.py' })).toBeInTheDocument();

    showVersion(
      createMockSkillVersion({
        version: 2,
        source_type: 'mlflow',
        source: `mlflow-artifacts:/${OTHER_ROOT}`,
        ref: null,
        subpath: null,
      }),
    );
    expect(await screen.findByText('10.0 MB')).toBeInTheDocument();
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  it('does not download a file too large to preview', async () => {
    renderFiles();

    await userEvent.click(await screen.findByText('huge.bin'));
    expect(await screen.findByText(/too large to preview here/)).toBeInTheDocument();
  });

  it.each([
    ['Git', createMockSkillVersion({ source_type: 'git', source: 'https://github.com/acme/skills', ref: 'main' })],
    ['OCI', createMockSkillVersion({ source_type: 'oci', source: 'ghcr.io/acme/skill:1', ref: null, subpath: null })],
  ])('renders nothing for a %s version, whose source is shown under Details', (_, version) => {
    const { container } = renderFiles(version);

    expect(container).toBeEmptyDOMElement();
  });

  describe('SkillManifestPreview', () => {
    const renderPreview = (version = uploaded) => {
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      return render(
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <QueryClientProvider client={client}>
              <SkillManifestPreview version={version} />
            </QueryClientProvider>
          </DesignSystemProvider>
        </IntlProvider>,
      );
    };

    it('renders the stored SKILL.md with its name and description above the instructions', async () => {
      renderPreview();

      expect(await screen.findByRole('heading', { name: 'Code review' })).toBeInTheDocument();
      expect(screen.getByText('SKILL.md')).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Copy SKILL.md' })).toBeInTheDocument();
      expect(screen.getByText('Skill name')).toBeInTheDocument();
      expect(screen.getByText('code-review')).toBeInTheDocument();
      expect(screen.getByText('Description')).toBeInTheDocument();
      expect(screen.getByText('Reviews a diff for bugs.')).toBeInTheDocument();
      expect(screen.queryByText(/^name:/)).not.toBeInTheDocument();
    });

    it('offers Show more once the preview grows past its collapsed height', async () => {
      const callbacks: ResizeObserverCallback[] = [];
      const resizeObserver = jest.spyOn(window, 'ResizeObserver').mockImplementation((callback) => {
        callbacks.push(callback);
        return { observe: jest.fn(), unobserve: jest.fn(), disconnect: jest.fn() };
      });
      const scrollHeight = jest.spyOn(Element.prototype, 'scrollHeight', 'get').mockReturnValue(100);
      try {
        renderPreview();

        expect(await screen.findByRole('heading', { name: 'Code review' })).toBeInTheDocument();
        expect(screen.queryByRole('button', { name: 'Show more' })).not.toBeInTheDocument();

        // A narrower page rewraps the markdown, so the same text gets taller.
        scrollHeight.mockReturnValue(1000);
        act(() => callbacks.forEach((callback) => callback([], {} as ResizeObserver)));
        expect(screen.getByRole('button', { name: 'Show more' })).toBeInTheDocument();
      } finally {
        scrollHeight.mockRestore();
        resizeObserver.mockRestore();
      }
    });

    it('expands when keyboard focus moves into the clipped part', async () => {
      const scrollHeight = jest.spyOn(Element.prototype, 'scrollHeight', 'get').mockReturnValue(1000);
      try {
        renderPreview();

        expect(await screen.findByRole('button', { name: 'Show more' })).toBeInTheDocument();
        const guide = screen.getByRole('link', { name: 'guide' });
        while (document.activeElement !== guide) await userEvent.tab();
        expect(screen.getByRole('button', { name: 'Show less' })).toBeInTheDocument();
      } finally {
        scrollHeight.mockRestore();
      }
    });

    it('renders nothing for a version without a stored SKILL.md', async () => {
      const { container } = renderPreview(
        createMockSkillVersion({ source_type: 'mlflow', source: `mlflow-artifacts:/${OTHER_ROOT}`, ref: null }),
      );

      // The listing has no SKILL.md, so there is nothing to fetch or show once it loads.
      await screen.findByText((_, element) => element === container);
      expect(container).toBeEmptyDOMElement();
    });

    it('renders nothing for a remote version, whose content the registry never fetches', () => {
      const { container } = renderPreview(
        createMockSkillVersion({ source_type: 'git', source: 'https://github.com/acme/skills', ref: 'main' }),
      );

      expect(container).toBeEmptyDOMElement();
    });
  });
});
