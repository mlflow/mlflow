import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import { IntlProvider } from 'react-intl';
import type { ExperimentSettingsNameInputProps } from './ExperimentSettingsNameInput';
import { ExperimentSettingsNameInput } from './ExperimentSettingsNameInput';

const TestProviders = ({ children }: React.PropsWithChildren<unknown>) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

const renderNameInput = (props: Partial<ExperimentSettingsNameInputProps> = {}) => {
  const onSave = props.onSave ?? jest.fn<ExperimentSettingsNameInputProps['onSave']>().mockResolvedValue(undefined);
  render(
    <>
      <span id="experiment-name-label">Experiment name</span>
      <ExperimentSettingsNameInput
        initialName="Original name"
        isEditable
        labelId="experiment-name-label"
        onSave={onSave}
        {...props}
      />
    </>,
    { wrapper: TestProviders },
  );
  return { onSave };
};

describe('ExperimentSettingsNameInput', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('saves the latest draft once after 500 ms of inactivity', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('Updated');
    await user.paste(' name');
    act(() => jest.advanceTimersByTime(499));
    expect(onSave).not.toHaveBeenCalled();

    act(() => jest.advanceTimersByTime(1));
    await act(async () => Promise.resolve());
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith('Updated name');
  });

  it('flushes an edited name on blur', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('Blurred name');
    await user.tab();
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith('Blurred name');
  });

  it('retains a failed draft and retries it successfully', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const onSave = jest
      .fn<ExperimentSettingsNameInputProps['onSave']>()
      .mockRejectedValueOnce(new Error('rename failed'))
      .mockResolvedValue(undefined);
    renderNameInput({ onSave });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('Retained draft');
    act(() => jest.advanceTimersByTime(500));
    await act(async () => Promise.resolve());

    expect(input).toHaveValue('Retained draft');
    expect(input).toHaveAccessibleDescription('Unable to save the experiment name. Retry');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenCalledTimes(2);
    expect(onSave).toHaveBeenLastCalledWith('Retained draft');
    expect(screen.queryByText('Unable to save the experiment name.')).not.toBeInTheDocument();
  });

  it('requires a non-empty name without offering a retry', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    act(() => jest.advanceTimersByTime(500));

    expect(input).toHaveAccessibleDescription('Experiment name is required.');
    expect(onSave).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'Retry' })).not.toBeInTheDocument();
  });
});
