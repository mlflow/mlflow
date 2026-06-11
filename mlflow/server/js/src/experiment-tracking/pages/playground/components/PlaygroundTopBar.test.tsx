import { jest, describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ToolChoice } from '../types';

jest.mock('../../../components/EndpointSelector', () => ({
  EndpointSelector: ({
    currentEndpointName,
    onEndpointSelect,
    showCreateButton,
  }: {
    currentEndpointName?: string;
    onEndpointSelect: (name: string) => void;
    showCreateButton?: boolean;
  }) => (
    <>
      <input
        data-testid="endpoint-selector-test-input"
        value={currentEndpointName ?? ''}
        onChange={(event) => onEndpointSelect(event.target.value)}
      />
      {showCreateButton && <div data-testid="endpoint-selector-create-button-enabled" />}
    </>
  ),
}));

// Imported after the mock so the mock takes effect.
// eslint-disable-next-line import/first
import { PlaygroundTopBar } from './PlaygroundTopBar';

const renderTopBar = () => {
  const onEndpointSelect = jest.fn();
  const onOpenRegistry = jest.fn();
  const noop = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <PlaygroundTopBar
          endpointName=""
          onEndpointSelect={onEndpointSelect}
          params={{}}
          onParamsChange={noop}
          toolsText=""
          onToolsChange={noop}
          toolsError={null}
          toolChoice={'none' as ToolChoice}
          onToolChoiceChange={noop}
          responseFormatType="text"
          onResponseFormatTypeChange={noop}
          responseFormatSchemaText=""
          onResponseFormatSchemaChange={noop}
          responseFormatSchemaError={null}
          messages={[]}
          variables={{}}
          onVariablesChange={noop}
          onOpenRegistry={onOpenRegistry}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onEndpointSelect, onOpenRegistry };
};

describe('PlaygroundTopBar', () => {
  it('renders the endpoint selector and the three top-bar buttons', () => {
    renderTopBar();
    expect(screen.getByTestId('endpoint-selector-test-input')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open model parameters/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open variable values/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /load prompt from registry/i })).toBeInTheDocument();
  });

  it('wires onOpenRegistry to the Load button', async () => {
    const { onOpenRegistry } = renderTopBar();
    await userEvent.click(screen.getByRole('button', { name: /load prompt from registry/i }));
    expect(onOpenRegistry).toHaveBeenCalledTimes(1);
  });

  it('enables the create-endpoint button on the endpoint selector', () => {
    renderTopBar();
    expect(screen.getByTestId('endpoint-selector-create-button-enabled')).toBeInTheDocument();
  });
});
