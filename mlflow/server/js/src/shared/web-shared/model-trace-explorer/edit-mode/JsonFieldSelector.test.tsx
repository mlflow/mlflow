import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { JsonFieldSelector } from './JsonFieldSelector';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('JsonFieldSelector', () => {
  it('renders top-level keys as checkboxes', () => {
    const data = { query: 'hello', max_results: 10 };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath={null}
        onPathChange={jest.fn()}
        label="Input Fields"
      />,
      { wrapper },
    );
    expect(screen.getByText('query')).toBeInTheDocument();
    expect(screen.getByText('max_results')).toBeInTheDocument();
  });

  it('renders nested objects as expandable nodes', async () => {
    const data = { config: { temperature: 0.7, model: 'gpt-4' } };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath={null}
        onPathChange={jest.fn()}
        label="Input Fields"
      />,
      { wrapper },
    );
    expect(screen.getByText('config')).toBeInTheDocument();
    expect(screen.queryByText('temperature')).not.toBeInTheDocument();
    await userEvent.click(screen.getByText('config'));
    expect(screen.getByText('temperature')).toBeInTheDocument();
    expect(screen.getByText('model')).toBeInTheDocument();
  });

  it('clicking a leaf checkbox calls onPathChange with JSONPath', async () => {
    const onPathChange = jest.fn();
    const data = { query: 'hello', context: 'world' };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath={null}
        onPathChange={onPathChange}
        label="Input Fields"
      />,
      { wrapper },
    );
    await userEvent.click(screen.getByRole('checkbox', { name: /query/ }));
    expect(onPathChange).toHaveBeenCalledWith('$.query');
  });

  it('renders arrays with index notation', async () => {
    const data = { messages: ['hello', 'world'] };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath={null}
        onPathChange={jest.fn()}
        label="Output Fields"
      />,
      { wrapper },
    );
    await userEvent.click(screen.getByText('messages'));
    expect(screen.getByText('[0]')).toBeInTheDocument();
    expect(screen.getByText('[1]')).toBeInTheDocument();
  });

  it('shows preview values for leaf nodes', () => {
    const data = { query: 'what is MLflow?', count: 42 };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath={null}
        onPathChange={jest.fn()}
        label="Input Fields"
      />,
      { wrapper },
    );
    expect(screen.getByText(/"what is MLflow\?"/)).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it('pre-checks field matching selectedPath', () => {
    const data = { query: 'hello', context: 'world' };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath="$.query"
        onPathChange={jest.fn()}
        label="Input Fields"
      />,
      { wrapper },
    );
    const checkbox = screen.getByRole('checkbox', { name: /query/ });
    expect(checkbox).toBeChecked();
  });

  it('shows generated JSONPath in read-only text field', () => {
    const data = { query: 'hello' };
    render(
      <JsonFieldSelector
        data={data}
        selectedPath="$.query"
        onPathChange={jest.fn()}
        label="Input Fields"
      />,
      { wrapper },
    );
    const pathField = screen.getByDisplayValue('$.query');
    expect(pathField).toBeInTheDocument();
  });
});
