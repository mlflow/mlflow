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
  toolAdded?: boolean;
  toolChoice?: ToolChoice;
  toolsText?: string;
  toolsError?: string | null;
  params?: PlaygroundParams;
  responseFormatType?: ResponseFormatType;
}

const renderButton = ({
  toolAdded = false,
  toolChoice = 'auto',
  toolsText = '',
  toolsError = null,
  params = {},
  responseFormatType = 'text',
}: RenderProps = {}) => {
  const onChange = jest.fn();
  const onToolsChange = jest.fn();
  const onAddTool = jest.fn<() => void>();
  const onRemoveTool = jest.fn<() => void>();
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
          toolAdded={toolAdded}
          onAddTool={onAddTool}
          onRemoveTool={onRemoveTool}
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
  return { onChange, onToolsChange, onAddTool, onRemoveTool, onToolChoiceChange, onResponseFormatTypeChange };
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

  it('shows only the Add tool button by default and hides the picker and JSON textarea', async () => {
    renderButton({ toolAdded: false });
    await openDrawer();
    expect(screen.getByRole('button', { name: 'Add tool' })).toBeInTheDocument();
    expect(screen.queryByRole('radio', { name: 'Auto' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('JSON Tool Definition')).not.toBeInTheDocument();
  });

  it('fires onAddTool when the Add tool button is clicked', async () => {
    const { onAddTool } = renderButton({ toolAdded: false });
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: 'Add tool' }));
    expect(onAddTool).toHaveBeenCalledTimes(1);
  });

  it('shows the JSON textarea plus an Auto/Required picker once a tool is added', async () => {
    renderButton({ toolAdded: true, toolChoice: 'auto' });
    await openDrawer();
    expect(screen.getByLabelText('JSON Tool Definition')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Auto' })).toBeChecked();
    expect(screen.getByRole('radio', { name: 'Required' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Add tool' })).not.toBeInTheDocument();
  });

  it('fires onRemoveTool when the Remove tool button is clicked', async () => {
    const { onRemoveTool } = renderButton({ toolAdded: true });
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: 'Remove tool' }));
    expect(onRemoveTool).toHaveBeenCalledTimes(1);
  });

  it('describes the add-tool flow in the Tools info popover', async () => {
    renderButton();
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: /about tool definitions/i }));
    expect(screen.getByText(/Click ‘Add tool’ to define a tool/i)).toBeInTheDocument();
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
