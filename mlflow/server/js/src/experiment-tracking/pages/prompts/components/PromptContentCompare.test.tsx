import { render, screen } from '@testing-library/react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { QueryClient, QueryClientProvider } from '@mlflow/mlflow/src/common/utils/reactQueryHooks';
import { PromptContentCompare } from './PromptContentCompare';

describe('PromptContentCompare', () => {
  const renderComponent = (baseline: any, compared: any) => {
    const queryClient = new QueryClient();
    return render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <QueryClientProvider client={queryClient}>
            <PromptContentCompare
              baselineVersion={baseline}
              comparedVersion={compared}
              onSwitchSides={jest.fn()}
              onEditVersion={jest.fn()}
              aliasesByVersion={{}}
            />
          </QueryClientProvider>
        </DesignSystemProvider>
      </IntlProvider>,
    );
  };

  it('stringifies chat prompts for comparison', () => {
    const baseline = {
      name: 'p',
      version: '1',
      tags: [{ key: 'mlflow.prompt.text', value: JSON.stringify([{ role: 'user', content: 'Hi' }]) }],
    };
    const compared = {
      name: 'p',
      version: '2',
      tags: [{ key: 'mlflow.prompt.text', value: JSON.stringify([{ role: 'user', content: 'Hi there' }]) }],
    };
    renderComponent(baseline, compared);
    expect(screen.getAllByText('user: Hi').length).toBeGreaterThan(0);
    expect(screen.getByText('there')).toBeInTheDocument();
  });
});
