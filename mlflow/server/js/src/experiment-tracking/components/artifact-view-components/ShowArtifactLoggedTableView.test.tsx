import userEvent from '@testing-library/user-event-14';
import { render, screen, waitFor } from '../../../common/utils/TestUtils.react18';
import { ShowArtifactLoggedTableView } from './ShowArtifactLoggedTableView';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

jest.setTimeout(30000); // Larger timeout for integration testing (table rendering)

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
  getArtifactContent: jest.fn().mockImplementation(() => Promise.resolve(JSON.stringify(testArtifactData))),
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
});
