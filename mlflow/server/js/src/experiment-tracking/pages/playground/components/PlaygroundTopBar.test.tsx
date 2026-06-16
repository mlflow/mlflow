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
  }: {
    currentEndpointName?: string;
    onEndpointSelect: (name: string) => void;
  }) => (
    <input
      data-testid="endpoint-selector-test-input"
      value={currentEndpointName ?? ''}
      onChange={(event) => onEndpointSelect(event.target.value)}
    />
  ),
}));

// Imported after the mock so the mock takes effect.
// eslint-disable-next-line import/first
import { PlaygroundTopBar } from './PlaygroundTopBar';

const renderTopBar = () => {
  const onEndpointSelect = jest.fn();
  const onOpenRegistry = jest.fn();
  const onOpenSave = jest.fn();
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
          onOpenSave={onOpenSave}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onEndpointSelect, onOpenRegistry, onOpenSave };
};

describe('PlaygroundTopBar', () => {
  it('renders the endpoint selector and the four top-bar buttons', () => {
    renderTopBar();
    expect(screen.getByTestId('endpoint-selector-test-input')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open model parameters/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open variable values/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /load prompt/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save prompt/i })).toBeInTheDocument();
  });

  it('wires onOpenRegistry to the Load button', async () => {
    const { onOpenRegistry } = renderTopBar();
    await userEvent.click(screen.getByRole('button', { name: /load prompt/i }));
    expect(onOpenRegistry).toHaveBeenCalledTimes(1);
  });

  it('wires onOpenSave to the Save button', async () => {
    const { onOpenSave } = renderTopBar();
    await userEvent.click(screen.getByRole('button', { name: /save prompt/i }));
    expect(onOpenSave).toHaveBeenCalledTimes(1);
  });
});
