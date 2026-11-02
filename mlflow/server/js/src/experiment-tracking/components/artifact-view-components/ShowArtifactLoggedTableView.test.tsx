import { jest, describe, beforeEach, beforeAll, afterAll, afterEach, it, expect } from '@jest/globals';
import { ErrorWrapper } from '../../../common/utils/ErrorWrapper';
import userEvent from '@testing-library/user-event';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { ShowArtifactLoggedTableView } from './ShowArtifactLoggedTableView';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { ArtifactPreviewTooLargeError, getArtifactBlob, getArtifactContent } from '../../../common/utils/ArtifactUtils';

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
jest.setTimeout(90000); // Larger timeout for integration testing (table rendering)

const testArtifactData = {
  columns: ['timestamp', 'level', 'message'],
  data: [
    ['row-1', 'INFO', 'Hello'],
    ['row-2', 'ERROR', 'Lorem ipsum'],
    ['row-3', 123, { key: 'value' }],
    ['row-4', false, null],
    ['row-5', true, [{ key: 'value' }]],
  ],
};

jest.mock('../../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../../common/utils/ArtifactUtils')>('../../../common/utils/ArtifactUtils'),
  getArtifactBlob: jest.fn(),
  getArtifactContent: jest.fn(),
}));

describe('ShowArtifactLoggedTableView', () => {
  const originalRevokeObjectURL = (URL as any).revokeObjectURL;
  const renderComponent = () => {
    render(
      <ShowArtifactLoggedTableView
        runUuid="test-run-uuid"
        path="/path/to/artifact"
        artifactRootUri="mlflow-artifacts:/1/test-run-uuid/artifacts"
      />,
      {
        wrapper: ({ children }) => (
          <IntlProvider locale="en">
            <DesignSystemProvider>{children}</DesignSystemProvider>
          </IntlProvider>
        ),
      },
    );
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Mock getBoundingClientRect to satisfy the table row layout
    jest.spyOn(window.Element.prototype, 'getBoundingClientRect').mockImplementation(
      () =>
        ({
          width: 1000,
          height: 1000,
          top: 0,
          right: 0,
          bottom: 0,
          left: 0,
        }) as DOMRect,
    );
  });

  let originalImageSrc: any;

  beforeAll(() => {
    // Mock <img> src setter to trigger load callback
    originalImageSrc = Object.getOwnPropertyDescriptor(window.Image.prototype, 'src');
    Object.defineProperty(window.Image.prototype, 'src', {
      set() {
        setTimeout(() => this.onload?.());
      },
      get() {},
    });
    (URL as any).revokeObjectURL = jest.fn();
  });

  afterAll(() => {
    Object.defineProperty(window.Image.prototype, 'src', originalImageSrc);
    (URL as any).revokeObjectURL = originalRevokeObjectURL;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it.each([
    ['preview size', new ArtifactPreviewTooLargeError(), /Maximum file size for preview: 50 MiB/],
    ['server mode', new ErrorWrapper({ artifacts_presigned_only: true }, 403), /Reload this page to continue/],
    [
      'artifact origin',
      new ErrorWrapper({ external_artifact_service: true, message: 'Use a same-origin artifact service.' }, 403),
      /Use a same-origin artifact service/,
    ],
    [
      'artifact root',
      new ErrorWrapper(
        { invalid_artifact_root: true, message: 'The artifact root URI cannot be decoded safely.' },
        400,
      ),
      /The artifact root URI cannot be decoded safely/,
    ],
  ])('shows actionable %s guidance', async (_name, error, guidance) => {
    jest.mocked(getArtifactContent).mockRejectedValue(error);
    renderComponent();
    expect(await screen.findByText(guidance as RegExp)).toBeInTheDocument();
  });

  it('renders the table and expected values in the cells', async () => {
    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testArtifactData)));

    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'timestamp' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'level' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'message' })).toBeInTheDocument();
    });

    // Wait for the table cells to render
    await waitFor(() => {
      for (const row of testArtifactData.data) {
        for (const cellValue of row) {
          const expectedValue = typeof cellValue === 'string' ? cellValue : JSON.stringify(cellValue);
          expect(screen.getByRole('cell', { name: expectedValue })).toBeInTheDocument();
        }
      }
    });
  });

  it('renders the table with columns and allows showing/hiding them', async () => {
    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testArtifactData)));

    renderComponent();

    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'timestamp' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'level' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'message' })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('button', { name: 'Table settings' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'level' }));
    await userEvent.click(screen.getByRole('menuitemcheckbox', { name: 'message' }));

    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'timestamp' })).toBeInTheDocument();
      expect(screen.queryByRole('columnheader', { name: 'level' })).not.toBeInTheDocument();
      expect(screen.queryByRole('columnheader', { name: 'message' })).not.toBeInTheDocument();
    });
  });

  it('renders the table of mixed types', async () => {
    const testMixedTypes = {
      columns: ['mixed_type_column'],
      data: [[1], ['two'], [3.0], [null], [[1, null, 3]], [{ 1: null, 3: 5 }]],
    };

    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testMixedTypes)));

    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'mixed_type_column' })).toBeInTheDocument();
    });

    // Wait for the table cells to render
    await waitFor(() => {
      for (const row of testMixedTypes.data) {
        for (const cellValue of row) {
          const expectedValue = typeof cellValue === 'string' ? cellValue : JSON.stringify(cellValue);
          expect(screen.getByRole('cell', { name: expectedValue })).toBeInTheDocument();
        }
      }
    });
  });

  it('renders table with mixed column types', async () => {
    const testMixedColumnTypes = {
      columns: [1, 'test', null, true],
      data: [
        [1, 'test', null, true],
        [2, 'test2', { key: 'value2' }, false],
      ],
    };

    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testMixedColumnTypes)));

    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      for (const column of testMixedColumnTypes.columns) {
        const columnName = typeof column === 'string' ? column : JSON.stringify(column);
        expect(screen.getByRole('columnheader', { name: columnName })).toBeInTheDocument();
      }
    });

    // Wait for the table cells to render
    await waitFor(() => {
      for (const row of testMixedColumnTypes.data) {
        for (const cellValue of row) {
          const expectedValue = typeof cellValue === 'string' ? cellValue : JSON.stringify(cellValue);
          expect(screen.getByRole('cell', { name: expectedValue })).toBeInTheDocument();
        }
      }
    });
  });

  it('renders table with empty data', async () => {
    const testEmptyData = {
      columns: ['column1', 'column2'],
      data: [],
    };

    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testEmptyData)));
    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'column1' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'column2' })).toBeInTheDocument();
    });

    // Wait for the table cells to render
    await waitFor(() => {
      expect(screen.queryByRole('cell')).not.toBeInTheDocument();
    });
  });

  it('renders table with empty columns', async () => {
    const testEmptyTable = {
      columns: [],
      data: [],
    };

    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testEmptyTable)));
    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      expect(screen.queryByRole('columnheader')).not.toBeInTheDocument();
    });

    // Wait for the table cells to render
    await waitFor(() => {
      expect(screen.queryByRole('cell')).not.toBeInTheDocument();
    });
  });

  it('renders table with image', async () => {
    const testImageTable = {
      columns: ['images'],
      data: [
        [
          {
            type: 'image',
            filepath: 'fakePathUncompressed',
            compressed_filepath: 'fakePath',
          },
        ],
      ],
    };

    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(testImageTable)));
    jest.mocked(getArtifactBlob).mockResolvedValue(new Blob(['image']));
    jest.spyOn(URL, 'createObjectURL').mockReturnValue('blob:artifact-image');
    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'images' })).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(getArtifactBlob).toHaveBeenCalledWith(
        'ajax-api/2.0/mlflow-artifacts/artifacts/1/test-run-uuid/artifacts/fakePath',
      );
    });

    // Wait for the table cells to render
    await waitFor(() => {
      const images = screen.getAllByRole('img');
      expect(images.length).toBeGreaterThan(0);
      expect(images.some((image) => image.getAttribute('src') === 'blob:artifact-image')).toBe(true);
    });
    expect(getArtifactBlob).not.toHaveBeenCalledWith(expect.stringContaining('/fakePathUncompressed'));
    const thumbnail = screen.getAllByRole('img').find((image) => image.getAttribute('src') === 'blob:artifact-image');
    await userEvent.click(thumbnail!);
    await waitFor(() =>
      expect(getArtifactBlob).toHaveBeenCalledWith(
        'ajax-api/2.0/mlflow-artifacts/artifacts/1/test-run-uuid/artifacts/fakePathUncompressed',
      ),
    );
  });

  it.each([
    { artifactContent: ['definitely-not-object-with-columns-and-data'], testName: 'non-standard array' },
    { artifactContent: { data: [] }, testName: 'object without columns' },
    { artifactContent: { columns: [] }, testName: 'object without data' },
  ])('renders error message when artifact contents is a $testName', async ({ artifactContent }) => {
    jest.mocked(getArtifactContent).mockImplementation(() => Promise.resolve(JSON.stringify(artifactContent)));

    renderComponent();

    // Wait for the render error message
    await waitFor(() => {
      expect(
        screen.getByText(
          "Unable to parse JSON file. The file should contain an object with 'columns' and 'data' keys.",
        ),
      ).toBeInTheDocument();
    });
  });

  it('renders fallback column headers when column names are not provided', async () => {
    jest.mocked(getArtifactContent).mockImplementation(() =>
      Promise.resolve(
        JSON.stringify({
          columns: ['guid', '', 'sentence', ''],
          data: [
            ['123', 'value_in_unnamed_column', 'Nostrud eu velit', 'value_in_unnamed_column'],
            ['345', 'value_in_unnamed_column', 'Excepteur consectetur', 'value_in_unnamed_column'],
          ],
        }),
      ),
    );

    renderComponent();

    // Autogenerated fallback column headers should appear
    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'column-2' })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: 'column-4' })).toBeInTheDocument();
    });
  });
});
