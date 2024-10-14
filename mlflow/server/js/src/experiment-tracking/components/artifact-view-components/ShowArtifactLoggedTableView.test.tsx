import userEvent from '@testing-library/user-event-14';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { ShowArtifactLoggedTableView } from './ShowArtifactLoggedTableView';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { getArtifactContent } from '../../../common/utils/ArtifactUtils';

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
  ...jest.requireActual('../../../common/utils/ArtifactUtils'),
  getArtifactContent: jest.fn(),
}));

describe('ShowArtifactLoggedTableView', () => {
  const renderComponent = () => {
    render(<ShowArtifactLoggedTableView runUuid="test-run-uuid" path="/path/to/artifact" />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>{children}</DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };

  beforeEach(() => {
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
        } as DOMRect),
    );
  });

  afterEach(() => {
    jest.restoreAllMocks();
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
    renderComponent();

    // Wait for the table headers to render
    await waitFor(() => {
      expect(screen.getByRole('columnheader', { name: 'images' })).toBeInTheDocument();
    });

    // Wait for the table cells to render
    await waitFor(() => {
      const image = screen.getByRole('img');
      expect(image).toBeInTheDocument();
      expect(image).toHaveAttribute(
        'src',
        expect.stringContaining('get-artifact?path=fakePath&run_uuid=test-run-uuid'),
      );
    });
  });
});
