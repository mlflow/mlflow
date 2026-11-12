import { jest, describe, it, expect, afterEach } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';

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
  const onOpenAddToDataset = jest.fn();
  const noop = jest.fn();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <PlaygroundTopBar
          endpointName=""
          onEndpointSelect={onEndpointSelect}
          params={{}}
          onParamsChange={noop}
          tools={[]}
          onAddTool={noop}
          onRemoveTool={noop}
          onUpdateTool={noop}
          toolChoice="auto"
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
          onOpenAddToDataset={onOpenAddToDataset}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onEndpointSelect, onOpenRegistry, onOpenSave, onOpenAddToDataset };
};

describe('PlaygroundTopBar', () => {
  it('renders the endpoint selector and the five top-bar buttons', () => {
    renderTopBar();
    expect(screen.getByTestId('endpoint-selector-test-input')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open model parameters/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /open variable values/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /load prompt/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /save prompt/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add to datasets/i })).toBeInTheDocument();
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

  it('wires onOpenAddToDataset to the Add to dataset button', async () => {
    const { onOpenAddToDataset } = renderTopBar();
    await userEvent.click(screen.getByRole('button', { name: /add to datasets/i }));
    expect(onOpenAddToDataset).toHaveBeenCalledTimes(1);
  });

  describe('prompt action labels', () => {
    const PROMPT_ACTIONS = ['Load prompt', 'Save prompt', 'Add to datasets'];

    // jsdom has no layout, so stand in for the measured top-bar width.
    const renderAtBarWidth = (width: number) => {
      jest.spyOn(HTMLElement.prototype, 'clientWidth', 'get').mockReturnValue(width);
      renderTopBar();
    };

    afterEach(() => {
      jest.restoreAllMocks();
    });

    it('shows only the icons when the bar is too narrow for the labels', () => {
      renderAtBarWidth(800);
      for (const name of PROMPT_ACTIONS) {
        expect(screen.getByRole('button', { name })).toHaveTextContent(/^$/);
      }
    });

    it('shows the labels when the bar is wide enough', () => {
      renderAtBarWidth(1200);
      for (const name of PROMPT_ACTIONS) {
        expect(screen.getByRole('button', { name })).toHaveTextContent(name);
      }
    });
  });
});
