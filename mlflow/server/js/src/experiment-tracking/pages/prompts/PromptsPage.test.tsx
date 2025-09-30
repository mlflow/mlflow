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

// eslint-disable-next-line no-restricted-syntax -- TODO(FEINF-4392)
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
    await userEvent.type(screen.getByLabelText('Name:'), 'prompt7');
    await userEvent.type(screen.getByLabelText('Prompt:'), 'lorem ipsum');
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

  it('should create a new chat prompt version', async () => {
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
    await userEvent.type(screen.getByLabelText('Name:'), 'prompt8');
    await userEvent.click(screen.getByRole('radio', { name: 'Chat' }));

    const firstContent = document.querySelector('textarea[name="chatMessages.0.content"]') as HTMLTextAreaElement;
    await userEvent.type(firstContent, 'Hello');
    await userEvent.click(screen.getAllByRole('button', { name: 'Add message' })[0]);
    await userEvent.clear(screen.getAllByPlaceholderText('role')[1]);
    await userEvent.type(screen.getAllByPlaceholderText('role')[1], 'assistant');
    const secondContent = document.querySelector('textarea[name="chatMessages.1.content"]') as HTMLTextAreaElement;
    await userEvent.type(secondContent, 'Hi!');
    await userEvent.type(screen.getByLabelText('Commit message (optional):'), 'commit message');
    await userEvent.click(screen.getByText('Create'));

    const expectedMessages = [
      { role: 'user', content: 'Hello' },
      { role: 'assistant', content: 'Hi!' },
    ];

    await waitFor(() => {
      expect(createVersionSpy).toHaveBeenCalled();
    });

    const payload = createVersionSpy.mock.calls[0][0];
    expect(payload).toMatchObject({
      name: 'prompt8',
      description: 'commit message',
    });
    expect(payload.tags).toEqual(
      expect.arrayContaining([
        { key: 'mlflow.prompt.is_prompt', value: 'true' },
        { key: 'mlflow.prompt.text', value: JSON.stringify(expectedMessages) },
        { key: '_mlflow_prompt_type', value: 'chat' },
      ]),
    );
  });
});
