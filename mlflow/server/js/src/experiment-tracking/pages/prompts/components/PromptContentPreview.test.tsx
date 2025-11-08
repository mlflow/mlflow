import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PromptContentPreview } from './PromptContentPreview';

describe('PromptContentPreview', () => {
  const renderComponent = (promptVersion: any) => {
    const queryClient = new QueryClient();
    return render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <PromptContentPreview
              promptVersion={promptVersion}
              aliasesByVersion={{}}
              showEditPromptVersionMetadataModal={jest.fn()}
            />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );
  };

  it('renders chat messages', () => {
    const promptVersion = {
      name: 'prompt1',
      version: '1',
      tags: [
        {
          key: 'mlflow.prompt.text',
          value: JSON.stringify([
            { role: 'user', content: 'Hi' },
            { role: 'assistant', content: 'Hello' },
          ]),
        },
        {
          key: '_mlflow_prompt_type',
          value: 'chat',
        },
      ],
    };
    renderComponent(promptVersion);

    expect(screen.getByText('user', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('assistant', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Hi', { exact: false })).toBeInTheDocument();
    expect(screen.getByText('Hello', { exact: false })).toBeInTheDocument();
  });
});
