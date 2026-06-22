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
  toolChoice?: ToolChoice;
  toolsText?: string;
  toolsError?: string | null;
  toolsAdded?: boolean;
  params?: PlaygroundParams;
  responseFormatType?: ResponseFormatType;
}

const renderButton = ({
  toolChoice = 'auto',
  toolsText = '',
  toolsError = null,
  toolsAdded = false,
  params = {},
  responseFormatType = 'text',
}: RenderProps = {}) => {
  const onChange = jest.fn();
  const onToolsChange = jest.fn();
  const onToolsAddedChange = jest.fn<(next: boolean) => void>();
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
          onToolsAddedChange={onToolsAddedChange}
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
  return { onChange, onToolsChange, onToolsAddedChange, onToolChoiceChange, onResponseFormatTypeChange };
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

  it('shows the Add tools button and hides the JSON textarea before tools are added', async () => {
    renderButton({ toolsAdded: false });
    await openDrawer();
    expect(screen.getByRole('button', { name: 'Add tools' })).toBeInTheDocument();
    expect(screen.queryByLabelText('JSON tool definitions')).not.toBeInTheDocument();
  });

  it('fires onToolsAddedChange(true) when Add tools is clicked', async () => {
    const { onToolsAddedChange } = renderButton({ toolsAdded: false });
    await openDrawer();
    await userEvent.click(screen.getByRole('button', { name: 'Add tools' }));
    expect(onToolsAddedChange).toHaveBeenLastCalledWith(true);
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
