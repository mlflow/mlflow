import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { render, screen, waitFor, within } from '@testing-library/react';
import { setupServer } from '../../../common/utils/setup-msw';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import PromptsDetailsPage from './PromptsDetailsPage';
import {
  getFailedRegisteredPromptDetailsResponse,
  getMockedRegisteredPromptDeleteResponse,
  getMockedRegisteredPromptDetailsResponse,
  getMockedRegisteredPromptSetTagsResponse,
  getMockedRegisteredPromptSourceRunResponse,
  getMockedRegisteredPromptVersionsResponse,
} from './test-utils';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { getTableRowByCellText } from '@databricks/design-system/test-utils/rtl';
import { MockedReduxStoreProvider } from '../../../common/utils/TestUtils';
import { REGISTERED_PROMPT_SOURCE_RUN_IDS } from './utils';

jest.setTimeout(30000); // increase timeout due to heavier use of tables, modals and forms

describe('PromptsDetailsPage', () => {
  const server = setupServer(
    getMockedRegisteredPromptDetailsResponse('prompt1'),
    getMockedRegisteredPromptVersionsResponse('prompt1', 2),
  );

  beforeAll(() => {
    process.env['MLFLOW_USE_ABSOLUTE_AJAX_URLS'] = 'true';
    server.listen();
  });

  const renderTestComponent = () => {
    const queryClient = new QueryClient();
    render(<PromptsDetailsPage />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <DesignSystemProvider>
            <MockedReduxStoreProvider>
              <TestRouter
                routes={[
                  testRoute(
                    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>,
                    '/prompt/:promptName',
                  ),
                  testRoute(<div />, '*'),
                ]}
                initialEntries={['/prompt/prompt1']}
              />
            </MockedReduxStoreProvider>
          </DesignSystemProvider>
        </IntlProvider>
      ),
    });
  };
  it('should render basic prompt details', async () => {
    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'prompt1' })).toBeInTheDocument();
    });

    expect(screen.getByRole('status', { name: 'some_tag' })).toBeInTheDocument();
  });

  it("should preview prompt versions' contents, aliases and commit message", async () => {
    server.use(getMockedRegisteredPromptVersionsResponse('prompt1', 2));
    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'prompt1' })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('radio', { name: 'Preview' }));

    await userEvent.click(screen.getByRole('cell', { name: 'Version 2' }));
    expect(screen.getByText('content for prompt version 2')).toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'alias2' })).toBeInTheDocument();
    expect(screen.getByText('some commit message for version 2')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('cell', { name: 'Version 1' }));
    expect(screen.getByText('content of prompt version 1')).toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'alias1' })).toBeInTheDocument();
    expect(screen.getByText('some commit message for version 1')).toBeInTheDocument();
  });

  it("should compare prompt versions' contents", async () => {
    server.use(getMockedRegisteredPromptVersionsResponse('prompt1', 3));

    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'prompt1' })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByRole('radio', { name: 'Compare' }));

    const table = screen.getByLabelText('Prompt versions table');

    const rowForVersion3 = getTableRowByCellText(table, 'Version 3', { columnHeaderName: 'Version' });
    const rowForVersion2 = getTableRowByCellText(table, 'Version 2', { columnHeaderName: 'Version' });

    await userEvent.click(within(rowForVersion3).getByLabelText('Select as baseline version'));
    await userEvent.click(within(rowForVersion2).getByLabelText('Select as compared version'));

    // Mocked data contains following content for versions:
    // Version 1: content of prompt version 1
    // Version 2: content for prompt version 2

    // We set up expected diffs and assert that they are displayed correctly:
    const diffByWord = [['text', 'content'], ' ', ['of', 'for'], ' prompt version ', ['3', '2']].flat().join('');
    expect(document.body).toHaveTextContent(diffByWord);

    // Switch sides and expect the diff to change:
    await userEvent.click(screen.getByLabelText('Switch sides'));
    const diffByWordSwitched = [['content', 'text'], ' ', ['for', 'of'], ' prompt version ', ['2', '3']]
      .flat()
      .join('');
    expect(document.body).toHaveTextContent(diffByWordSwitched);
  });

  it('should edit tags', async () => {
    const setTagSpy = jest.fn();
    server.use(getMockedRegisteredPromptSetTagsResponse(setTagSpy));

    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'prompt1' })).toBeInTheDocument();
    });

    expect(screen.getByLabelText('Edit tags')).toBeInTheDocument();

    await userEvent.click(screen.getByLabelText('Edit tags'));

    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    await userEvent.type(screen.getByRole('combobox'), 'new_tag');
    await userEvent.click(screen.getByText('Add tag "new_tag"'));

    await userEvent.type(screen.getByPlaceholderText('Type a value'), 'new_value');
    await userEvent.click(screen.getByLabelText('Add tag'));

    await userEvent.click(screen.getByText('Save tags'));

    await waitFor(() => {
      expect(setTagSpy).toHaveBeenCalledWith({ key: 'new_tag', value: 'new_value', name: 'prompt1' });
    });
  });

  it('should delete the prompt', async () => {
    const deletePromptSpy = jest.fn();
    server.use(getMockedRegisteredPromptDeleteResponse(deletePromptSpy));

    renderTestComponent();

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'prompt1' })).toBeInTheDocument();
    });

    await userEvent.click(screen.getByLabelText('More actions'));
    await userEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Delete'));

    await waitFor(() => {
      expect(deletePromptSpy).toHaveBeenCalledWith({ name: 'prompt1' });
    });
  });

  it('should display table and react to change page mode', async () => {
    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'prompt1' })).toBeInTheDocument();
    });

    expect(screen.getByRole('cell', { name: 'Version 2' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Version 1' })).toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'Preview' }));
    expect(screen.queryByRole('columnheader', { name: 'Registered at' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'Compare' }));
    expect(screen.queryByRole('columnheader', { name: 'Registered at' })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole('radio', { name: 'List' }));
    expect(screen.getByRole('columnheader', { name: 'Registered at' })).toBeInTheDocument();
  });

  it('should display error boundary UI component upon showstopper failure', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});
    server.use(getFailedRegisteredPromptDetailsResponse(404));

    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByRole('heading', { name: 'Error' })).toBeInTheDocument();
    });
    expect(screen.getByText('The requested resource was not found.')).toBeInTheDocument();
    jest.restoreAllMocks();
  });
});
