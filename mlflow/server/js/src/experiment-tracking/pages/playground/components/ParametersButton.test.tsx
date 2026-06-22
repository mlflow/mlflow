import { jest, describe, it, expect } from '@jest/globals';
import { PointerEventsCheckLevel } from '@testing-library/user-event';
import { render, screen } from '@testing-library/react';
import userEventGlobal from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { PlaygroundParams, PlaygroundTool, ResponseFormatType, ToolChoice } from '../types';
import { ParametersButton } from './ParametersButton';

const userEvent = userEventGlobal.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });

interface RenderProps {
  toolChoice?: ToolChoice;
  tools?: PlaygroundTool[];
  params?: PlaygroundParams;
  responseFormatType?: ResponseFormatType;
}

const renderButton = ({
  toolChoice = 'auto',
  tools = [],
  params = {},
  responseFormatType = 'text',
}: RenderProps = {}) => {
  const onChange = jest.fn();
  const onAddTool = jest.fn<() => void>();
  const onRemoveTool = jest.fn<(id: string) => void>();
  const onUpdateTool = jest.fn<(id: string, patch: Partial<PlaygroundTool>) => void>();
  const onToolChoiceChange = jest.fn<(next: ToolChoice) => void>();
  const onResponseFormatTypeChange = jest.fn();
  const onResponseFormatSchemaChange = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ParametersButton
          value={params}
          onChange={onChange}
          tools={tools}
          onAddTool={onAddTool}
          onRemoveTool={onRemoveTool}
          onUpdateTool={onUpdateTool}
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
  return { onChange, onAddTool, onRemoveTool, onUpdateTool, onToolChoiceChange, onResponseFormatTypeChange };
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

  it('shows the Add tools button and no tool cards before any tools exist', async () => {
    renderButton({ tools: [] });
    await openDrawer();
    expect(screen.getByRole('button', { name: 'Add tools' })).toBeInTheDocument();
    expect(screen.queryByLabelText('Tool 1 parameters')).not.toBeInTheDocument();
  });

  it('fires onAddTool when Add tools is clicked', async () => {
    const { onAddTool } = renderButton({ tools: [] });
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: 'Add tools' }));
    expect(onAddTool).toHaveBeenCalledTimes(1);
  });

  it('renders a tool card with a parameters editor when a tool exists', async () => {
    renderButton({ tools: [{ id: 't1', name: 'get_weather', description: '', params: '{}' }] });
    await openDrawer();
    expect(screen.getByLabelText('Tool 1 parameters')).toBeInTheDocument();
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
