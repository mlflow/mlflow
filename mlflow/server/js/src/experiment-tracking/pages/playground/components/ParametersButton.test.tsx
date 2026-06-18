import { jest, describe, it, expect } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import { render, screen } from '@testing-library/react';
import userEventGlobal from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { PlaygroundParams, ResponseFormatType, ToolChoice } from '../types';
import { ParametersButton } from './ParametersButton';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

interface RenderProps {
  toolsAdded?: boolean;
  toolChoice?: ToolChoice;
  toolsText?: string;
  toolsError?: string | null;
  params?: PlaygroundParams;
  responseFormatType?: ResponseFormatType;
}

const renderButton = ({
  toolsAdded = false,
  toolChoice = 'auto',
  toolsText = '',
  toolsError = null,
  params = {},
  responseFormatType = 'text',
}: RenderProps = {}) => {
  const onChange = jest.fn();
  const onToolsChange = jest.fn();
  const onAddTools = jest.fn<() => void>();
  const onRemoveTools = jest.fn<() => void>();
  const onToolChoiceChange = jest.fn<(next: ToolChoice) => void>();
  const onResponseFormatTypeChange = jest.fn();
  const onResponseFormatSchemaChange = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ParametersButton
          value={params}
          onChange={onChange}
          toolsText={toolsText}
          onToolsChange={onToolsChange}
          toolsError={toolsError}
          toolsAdded={toolsAdded}
          onAddTools={onAddTools}
          onRemoveTools={onRemoveTools}
          toolChoice={toolChoice}
          onToolChoiceChange={onToolChoiceChange}
          responseFormatType={responseFormatType}
          onResponseFormatTypeChange={onResponseFormatTypeChange}
          responseFormatSchemaText=""
          onResponseFormatSchemaChange={onResponseFormatSchemaChange}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange, onToolsChange, onAddTools, onRemoveTools, onToolChoiceChange, onResponseFormatTypeChange };
};

const openDrawer = async () => {
  await userEvent.click(screen.getByRole('button', { name: /open model parameters/i }));
};

describe('ParametersButton', () => {
  it('opens the Settings drawer with the three section headers', async () => {
    renderButton();
    await openDrawer();
    expect(screen.getByText('Parameters')).toBeInTheDocument();
    expect(screen.getByText('Tools')).toBeInTheDocument();
    expect(screen.getByText('Response format')).toBeInTheDocument();
  });

  it('shows only the Add tools button by default and hides the picker and JSON textarea', async () => {
    renderButton({ toolsAdded: false });
    await openDrawer();
    expect(screen.getByRole('button', { name: 'Add tools' })).toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: 'Auto' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('JSON Tool Definitions')).not.toBeInTheDocument();
  });

  it('fires onAddTools when the Add tools button is clicked', async () => {
    const { onAddTools } = renderButton({ toolsAdded: false });
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: 'Add tools' }));
    expect(onAddTools).toHaveBeenCalledTimes(1);
  });

  it('shows the JSON textarea plus an Auto/Required picker once tools are added', async () => {
    renderButton({ toolsAdded: true, toolChoice: 'auto' });
    await openDrawer();
    expect(screen.getByLabelText('JSON Tool Definitions')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Auto' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Required' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add tools' })).not.toBeInTheDocument();
  });

  it('fires onRemoveTools when the Remove tools button is clicked', async () => {
    const { onRemoveTools } = renderButton({ toolsAdded: true });
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: 'Remove tools' }));
    expect(onRemoveTools).toHaveBeenCalledTimes(1);
  });

  it('describes the add-tools flow in the Tools info popover', async () => {
    renderButton();
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: /about tool definitions/i }));
    expect(screen.getByText(/Click ‘Add tools’ to provide/i)).toBeInTheDocument();
    expect(screen.queryByText(/never call a tool/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Pick Auto or Required/i)).not.toBeInTheDocument();
  });

  it('opens the Parameters info popover and shows the provider-disclaimer line', async () => {
    renderButton();
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: /about sampling parameters/i }));
    expect(
      screen.getByText(/Some parameters may not be supported by the provider you are using\./i),
    ).toBeInTheDocument();
  });
});
