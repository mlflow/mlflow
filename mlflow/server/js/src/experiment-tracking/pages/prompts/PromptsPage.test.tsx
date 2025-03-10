import { render, screen, waitFor } from '@testing-library/react';
import PromptsPage from './PromptsPage';
import { QueryClientProvider, QueryClient } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { setupServer } from '../../../common/utils/setup-msw';
import { IntlProvider } from 'react-intl';
import { setupTestRouter, testRoute, TestRouter } from '../../../common/utils/RoutingTestUtils';
import userEvent from '@testing-library/user-event';
import {
  getMockedRegisteredPromptCreateResponse,
  getMockedRegisteredPromptCreateVersionResponse,
  getMockedRegisteredPromptSetTagsResponse,
  getMockedRegisteredPromptsResponse,
  getMockedRegisteredPromptVersionSetTagsResponse,
} from './test-utils';

jest.setTimeout(30000); // increase timeout due to heavier use of tables, modals and forms

describe('PromptsPage', () => {
  const server = setupServer(getMockedRegisteredPromptsResponse(2));

  const renderTestComponent = () => {
    const queryClient = new QueryClient();
    render(<PromptsPage />, {
      wrapper: ({ children }) => (
        <IntlProvider locale="en">
          <TestRouter
            routes={[
              testRoute(<QueryClientProvider client={queryClient}>{children}</QueryClientProvider>, '/'),
              testRoute(<div />, '*'),
            ]}
            initialEntries={['/']}
          />
        </IntlProvider>
      ),
    });
  };
  it('should render table contents', async () => {
    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByText('Prompts')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'prompt1' })).toBeInTheDocument();
      expect(screen.getByRole('link', { name: 'prompt2' })).toBeInTheDocument();
    });

    expect(screen.getByRole('cell', { name: 'Version 3' })).toBeInTheDocument();
    expect(screen.getByRole('cell', { name: 'Version 5' })).toBeInTheDocument();

    expect(screen.getByRole('status', { name: 'some_tag' })).toBeInTheDocument();
    expect(screen.getByRole('status', { name: 'another_tag' })).toBeInTheDocument();
  });

  it('should edit tags', async () => {
    const setTagMock = jest.fn();
    server.use(getMockedRegisteredPromptsResponse(1), getMockedRegisteredPromptSetTagsResponse(setTagMock));

    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByText('Prompts')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'prompt1' })).toBeInTheDocument();
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
      expect(setTagMock).toHaveBeenCalledWith({ key: 'new_tag', value: 'new_value', name: 'prompt1' });
    });
  });

  it('should create a new prompt version', async () => {
    const createVersionSpy = jest.fn();
    server.use(
      getMockedRegisteredPromptsResponse(0),
      getMockedRegisteredPromptCreateResponse(),
      getMockedRegisteredPromptCreateVersionResponse(createVersionSpy),
    );

    renderTestComponent();
    await waitFor(() => {
      expect(screen.getByText('Prompts')).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText('Create prompt'));

    await waitFor(() => {
      expect(screen.getByRole('dialog')).toBeInTheDocument();
    });
    await userEvent.type(screen.getByLabelText('Name (required):'), 'prompt7');
    await userEvent.type(screen.getByLabelText('Prompt (required):'), 'lorem ipsum');
    await userEvent.type(screen.getByLabelText('Commit message (optional):'), 'commit message');
    await userEvent.click(screen.getByText('Create'));

    await waitFor(() => {
      expect(createVersionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'prompt7',
          description: 'commit message',
          tags: expect.arrayContaining([
            {
              key: 'mlflow.prompt.is_prompt',
              value: 'true',
            },
            {
              key: 'mlflow.prompt.text',
              value: 'lorem ipsum',
            },
          ]),
        }),
      );
    });
  });
});
