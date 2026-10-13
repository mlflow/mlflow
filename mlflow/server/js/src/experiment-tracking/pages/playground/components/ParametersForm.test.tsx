import { jest, describe, it, expect } from '@jest/globals';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import type { PlaygroundParams } from '../types';
import { ParametersForm } from './ParametersForm';

interface HarnessProps {
  initial: PlaygroundParams;
  onChange: (next: PlaygroundParams) => void;
}

// Wrap ParametersForm in a stateful harness so controlled Inputs see the
// updated value after each keystroke. Without this, userEvent.type would
// repeatedly type into a stale empty input.
const Harness = ({ initial, onChange }: HarnessProps) => {
  const [value, setValue] = useState<PlaygroundParams>(initial);
  return (
    <ParametersForm
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
    />
  );
};

const renderForm = (initial: PlaygroundParams = {}) => {
  const onChange = jest.fn<(next: PlaygroundParams) => void>();
  render(
    <IntlProvider locale="en">
      <DesignSystemProvider>
        <Harness initial={initial} onChange={onChange} />
      </DesignSystemProvider>
    </IntlProvider>,
  );
  return { onChange };
};

describe('ParametersForm', () => {
  it('forwards typed numeric values into the merged PlaygroundParams', async () => {
    const { onChange } = renderForm({ top_p: 0.5 });
    await userEvent.type(screen.getByLabelText('Temperature'), '1.2');
    // userEvent.type fires one onChange per keystroke; assert the final call captures the full value
    // and that other fields survive the merge.
    expect(onChange).toHaveBeenLastCalledWith({ top_p: 0.5, temperature: 1.2 });
  });

  it('clears a numeric input to undefined when the user empties it', () => {
    const { onChange } = renderForm({ temperature: 1.5 });
    const input = screen.getByLabelText('Temperature') as HTMLInputElement;
    fireEvent.change(input, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith({ temperature: undefined });
  });

  it('hides advanced inputs until the toggle is clicked', async () => {
    renderForm();
    expect(screen.queryByLabelText('Top K')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Stop sequences')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /advanced/i }));
    expect(screen.getByLabelText('Top K')).toBeInTheDocument();
    expect(screen.getByLabelText('Presence penalty')).toBeInTheDocument();
    expect(screen.getByLabelText('Frequency penalty')).toBeInTheDocument();
    expect(screen.getByLabelText('Stop sequences')).toBeInTheDocument();
  });

  it('parses stop sequences as a trimmed list of non-empty lines', async () => {
    const { onChange } = renderForm();
    await userEvent.click(screen.getByRole('button', { name: /advanced/i }));
    const stop = screen.getByLabelText('Stop sequences') as HTMLTextAreaElement;
    // fireEvent.change avoids userEvent's special handling of newlines.
    fireEvent.change(stop, { target: { value: 'foo\nbar\n\n  baz  ' } });
    expect(onChange).toHaveBeenLastCalledWith({ stop: ['foo', 'bar', 'baz'] });
  });

  it('returns undefined for stop when the textarea is cleared', async () => {
    const { onChange } = renderForm({ stop: ['x'] });
    await userEvent.click(screen.getByRole('button', { name: /advanced/i }));
    const stop = screen.getByLabelText('Stop sequences') as HTMLTextAreaElement;
    fireEvent.change(stop, { target: { value: '' } });
    expect(onChange).toHaveBeenLastCalledWith({ stop: undefined });
  });
});
