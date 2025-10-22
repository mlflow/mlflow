import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { OptimizeModal } from './OptimizeModal';

describe('OptimizeModal', () => {
  const renderComponent = (props: {
    visible: boolean;
    promptName: string;
    promptVersion: string;
    onCancel: jest.Mock;
  }) => {
    return render(
      <IntlProvider locale="en">
        <DesignSystemProvider>
          <OptimizeModal {...props} />
        </DesignSystemProvider>
      </IntlProvider>,
    );
  };

  it('renders modal when visible is true', () => {
    const onCancel = jest.fn();
    renderComponent({
      visible: true,
      promptName: 'my-prompt',
      promptVersion: '1',
      onCancel,
    });

    expect(screen.getByText('Optimize Prompt')).toBeInTheDocument();
    expect(
      screen.getByText("Here's how to optimize your prompt with your dataset in your Python code:"),
    ).toBeInTheDocument();
  });

  it('does not render modal when visible is false', () => {
    const onCancel = jest.fn();
    renderComponent({
      visible: false,
      promptName: 'my-prompt',
      promptVersion: '1',
      onCancel,
    });

    expect(screen.queryByText('Optimize Prompt')).not.toBeInTheDocument();
  });

  it('displays pip install command', () => {
    const onCancel = jest.fn();
    renderComponent({
      visible: true,
      promptName: 'my-prompt',
      promptVersion: '1',
      onCancel,
    });

    expect(screen.getByText(/pip install -U 'mlflow>=3.5.0' 'dspy>=3.0.0' openai/)).toBeInTheDocument();
  });

  it('displays Python code with interpolated prompt name and version', () => {
    const onCancel = jest.fn();
    renderComponent({
      visible: true,
      promptName: 'my-prompt',
      promptVersion: '2',
      onCancel,
    });

    expect(screen.getAllByText('prompts:/my-prompt/2', { exact: false })).toHaveLength(2);
  });

  it('calls onCancel when modal is closed', async () => {
    const onCancel = jest.fn();
    renderComponent({
      visible: true,
      promptName: 'my-prompt',
      promptVersion: '1',
      onCancel,
    });

    // Find and click the close button (X button in the modal header)
    const closeButton = screen.getByRole('button', { name: /close/i });
    await userEvent.click(closeButton);

    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
