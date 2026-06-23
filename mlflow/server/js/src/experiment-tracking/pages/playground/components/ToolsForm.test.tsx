import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { PlaygroundTool, ToolChoice } from '../types';
import { ToolsForm } from './ToolsForm';

// Monaco does not render in jsdom; stand the editor in with a labelled textarea.
jest.mock('../../experiment-evaluation-datasets-v2/components/LazyJsonRecordEditor', () => ({
  LazyJsonRecordEditor: ({
    ariaLabel,
    value,
    onChange,
    errorMessage,
  }: {
    ariaLabel: string;
    value: string;
    onChange: (next: string) => void;
    errorMessage?: string;
  }) => (
    <div>
      <textarea aria-label={ariaLabel} value={value} onChange={(event) => onChange(event.target.value)} />
      {errorMessage ? <div role="alert">{errorMessage}</div> : null}
    </div>
  ),
}));

interface RenderProps {
  tools?: PlaygroundTool[];
  toolChoice?: ToolChoice;
}

const tool = (overrides: Partial<PlaygroundTool> = {}): PlaygroundTool => ({
  id: 't1',
  name: 'get_weather',
  description: '',
  params: '{"type":"object","properties":{}}',
  ...overrides,
});

const renderForm = ({ tools = [], toolChoice = 'auto' }: RenderProps = {}) => {
  const onAddTool = jest.fn<() => void>();
  const onRemoveTool = jest.fn<(id: string) => void>();
  const onUpdateTool = jest.fn<(id: string, patch: Partial<PlaygroundTool>) => void>();
  const onToolChoiceChange = jest.fn<(next: ToolChoice) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <ToolsForm
          tools={tools}
          onAddTool={onAddTool}
          onRemoveTool={onRemoveTool}
          onUpdateTool={onUpdateTool}
          toolChoice={toolChoice}
          onToolChoiceChange={onToolChoiceChange}
        />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onAddTool, onRemoveTool, onUpdateTool, onToolChoiceChange };
};

describe('ToolsForm', () => {
  it('shows the Add tools button and no tool cards when there are no tools', () => {
    renderForm({ tools: [] });
    expect(screen.getByRole('button', { name: 'Add tools' })).toBeInTheDocument();
    expect(screen.queryByLabelText('Function name')).not.toBeInTheDocument();
  });

  it('fires onAddTool when the empty-state Add tools button is clicked', async () => {
    const { onAddTool } = renderForm({ tools: [] });
    await userEvent.click(screen.getByRole('button', { name: 'Add tools' }));
    expect(onAddTool).toHaveBeenCalledTimes(1);
  });

  it('renders name, description, parameters, the selector, and Add tool once a tool exists', () => {
    renderForm({ tools: [tool()] });
    expect(screen.getByLabelText('Function name')).toBeInTheDocument();
    expect(screen.getByLabelText('Function description')).toBeInTheDocument();
    expect(screen.getByLabelText('Tool 1 parameters')).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Auto' })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: 'Required' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add tool' })).toBeInTheDocument();
  });

  it('forwards picker changes via onToolChoiceChange', async () => {
    const { onToolChoiceChange } = renderForm({ tools: [tool()], toolChoice: 'auto' });
    await userEvent.click(screen.getByRole('radio', { name: 'Required' }));
    expect(onToolChoiceChange).toHaveBeenLastCalledWith('required');
  });

  it('forwards function name edits via onUpdateTool', () => {
    const { onUpdateTool } = renderForm({ tools: [tool({ name: '' })] });
    fireEvent.change(screen.getByLabelText('Function name'), { target: { value: 'lookup' } });
    expect(onUpdateTool).toHaveBeenLastCalledWith('t1', { name: 'lookup' });
  });

  it('forwards description edits via onUpdateTool', () => {
    const { onUpdateTool } = renderForm({ tools: [tool()] });
    fireEvent.change(screen.getByLabelText('Function description'), { target: { value: 'Looks things up' } });
    expect(onUpdateTool).toHaveBeenLastCalledWith('t1', { description: 'Looks things up' });
  });

  it('forwards parameters edits via onUpdateTool', () => {
    const { onUpdateTool } = renderForm({ tools: [tool()] });
    fireEvent.change(screen.getByLabelText('Tool 1 parameters'), { target: { value: '{"type":"object"}' } });
    expect(onUpdateTool).toHaveBeenLastCalledWith('t1', { params: '{"type":"object"}' });
  });

  it('fires onAddTool when the Add tool button is clicked', async () => {
    const { onAddTool } = renderForm({ tools: [tool()] });
    await userEvent.click(screen.getByRole('button', { name: 'Add tool' }));
    expect(onAddTool).toHaveBeenCalledTimes(1);
  });

  it('fires onRemoveTool with the tool id when its trash button is clicked', async () => {
    const { onRemoveTool } = renderForm({ tools: [tool()] });
    await userEvent.click(screen.getByRole('button', { name: 'Remove tool 1' }));
    expect(onRemoveTool).toHaveBeenLastCalledWith('t1');
  });

  it('does not show the required error before the function name is touched', () => {
    renderForm({ tools: [tool({ name: '' })] });
    expect(screen.queryByText('Function name is required')).not.toBeInTheDocument();
  });

  it('shows the required error after the function name is blurred while empty', () => {
    renderForm({ tools: [tool({ name: '' })] });
    fireEvent.blur(screen.getByLabelText('Function name'));
    expect(screen.getByText('Function name is required')).toBeInTheDocument();
  });

  it('shows a parameters error when the schema is not a JSON object', () => {
    renderForm({ tools: [tool({ params: '[]' })] });
    expect(screen.getByText('Parameters must be a JSON object')).toBeInTheDocument();
  });

  it('shows a parameters error when the schema is missing a properties map', () => {
    renderForm({ tools: [tool({ params: '{"type":"object"}' })] });
    expect(screen.getByText('Parameters schema must include a "properties" object')).toBeInTheDocument();
  });

  it('shows no errors for a named tool with a valid parameters object', () => {
    renderForm({ tools: [tool({ name: 'get_weather', params: '{"type":"object","properties":{}}' })] });
    expect(screen.queryByText('Function name is required')).not.toBeInTheDocument();
    expect(screen.queryByText('Parameters must be a JSON object')).not.toBeInTheDocument();
    expect(screen.queryByText('Parameters schema must include a "properties" object')).not.toBeInTheDocument();
  });

  it('numbers and labels multiple tool cards independently', () => {
    renderForm({ tools: [tool({ id: 't1' }), tool({ id: 't2' })] });
    expect(screen.getByLabelText('Tool 1 parameters')).toBeInTheDocument();
    expect(screen.getByLabelText('Tool 2 parameters')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Remove tool 2' })).toBeInTheDocument();
  });
});
