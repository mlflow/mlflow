import { afterEach, beforeEach, describe, expect, it, jest } from '@jest/globals';
import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DesignSystemProvider } from '@databricks/design-system';
import type { ReactNode } from 'react';
import { IntlProvider } from 'react-intl';
import type { ExperimentSettingsNameInputProps } from './ExperimentSettingsNameInput';
import { ExperimentSettingsNameInput } from './ExperimentSettingsNameInput';

const ProvidersWrapper = ({ children }: { children?: ReactNode }) => (
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
    { wrapper: ProvidersWrapper },
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

  it('saves the name without surrounding whitespace, and shows it trimmed after leaving the field', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('  Updated name  ');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith('Updated name');

    await act(async () => {
      input.blur();
    });
    expect(input).toHaveValue('Updated name');
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it('does not save a change that only adds surrounding whitespace', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.click(input);
    await user.paste('   ');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    await act(async () => {
      input.blur();
    });

    expect(onSave).not.toHaveBeenCalled();
    expect(input).toHaveValue('Original name');
  });

  it('treats a name of only spaces as empty', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('   ');
    await act(async () => {
      input.blur();
    });

    expect(await screen.findByText('Experiment name is required.')).toBeInTheDocument();
    expect(onSave).not.toHaveBeenCalled();
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

    await act(async () => {
      jest.advanceTimersByTime(1);
    });
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

  it('flushes an edited name on Enter', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput();
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('Entered name');
    await user.keyboard('{Enter}');
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenCalledTimes(1);
    expect(onSave).toHaveBeenCalledWith('Entered name');
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
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(input).toHaveValue('Retained draft');
    expect(input).toHaveAccessibleDescription('Unable to save the experiment name. Retry');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenCalledTimes(2);
    expect(onSave).toHaveBeenLastCalledWith('Retained draft');
    expect(screen.queryByText(/Unable to save the experiment name/)).not.toBeInTheDocument();
  });

  it('keeps the error and Retry when the name is edited back to the value that failed to save', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const onSave = jest
      .fn<ExperimentSettingsNameInputProps['onSave']>()
      .mockRejectedValueOnce(new Error('rename failed'))
      .mockResolvedValue(undefined);
    renderNameInput({ onSave });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('Failed name');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    expect(screen.getByText('Unable to save the experiment name.')).toBeInTheDocument();

    // Delete a character and type it again, then add a trailing space: the same name each time.
    await user.type(input, '{Backspace}e ');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    expect(input).toHaveValue('Failed name ');
    expect(screen.getByText('Unable to save the experiment name.')).toBeInTheDocument();
    expect(onSave).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await act(async () => Promise.resolve());
    expect(onSave).toHaveBeenCalledTimes(2);
    expect(onSave).toHaveBeenLastCalledWith('Failed name');
    expect(screen.queryByText(/Unable to save the experiment name/)).not.toBeInTheDocument();
  });

  it('clears the error once the name is changed to a different value, and saves that value', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const onSave = jest
      .fn<ExperimentSettingsNameInputProps['onSave']>()
      .mockRejectedValueOnce(new Error('rename failed'))
      .mockResolvedValue(undefined);
    renderNameInput({ onSave });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.click(input);
    await user.paste('Failed name');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    await user.type(input, '2');
    expect(screen.queryByText(/Unable to save the experiment name/)).not.toBeInTheDocument();
    await act(async () => {
      jest.advanceTimersByTime(500);
    });

    expect(onSave).toHaveBeenCalledTimes(2);
    expect(onSave).toHaveBeenLastCalledWith('Failed name2');
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

  it('keeps the autosave timer running when the page re-renders with a new save callback', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const firstSave = jest.fn<ExperimentSettingsNameInputProps['onSave']>().mockResolvedValue(undefined);
    const secondSave = jest.fn<ExperimentSettingsNameInputProps['onSave']>().mockResolvedValue(undefined);
    const nameInput = (onSave: ExperimentSettingsNameInputProps['onSave']) => (
      <>
        <span id="experiment-name-label">Experiment name</span>
        <ExperimentSettingsNameInput
          initialName="Original name"
          isEditable
          labelId="experiment-name-label"
          onSave={onSave}
        />
      </>
    );
    const { rerender } = render(nameInput(firstSave), { wrapper: ProvidersWrapper });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.paste('Updated name');
    act(() => jest.advanceTimersByTime(300));
    rerender(nameInput(secondSave));
    await act(async () => {
      jest.advanceTimersByTime(200);
    });

    expect([...firstSave.mock.calls, ...secondSave.mock.calls]).toEqual([['Updated name']]);
  });

  it("shows one message for a failed rename, not the server's error text", async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    const { onSave } = renderNameInput({
      onSave: jest
        .fn<ExperimentSettingsNameInputProps['onSave']>()
        .mockRejectedValue(new Error("Experiment 'Taken name' already exists.")),
    });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.paste('Taken name');
    await user.tab();
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenCalledWith('Taken name');
    expect(await screen.findByText('Unable to save the experiment name.')).toBeVisible();
    expect(screen.queryByText(/already exists/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeVisible();
  });

  it('saves the original name again when it is restored while another save is in flight', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    let finishFirstSave: () => void = () => undefined;
    const onSave = jest
      .fn<ExperimentSettingsNameInputProps['onSave']>()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            finishFirstSave = resolve;
          }),
      )
      .mockResolvedValue(undefined);
    const nameInput = (initialName: string) => (
      <>
        <span id="experiment-name-label">Experiment name</span>
        <ExperimentSettingsNameInput
          initialName={initialName}
          isEditable
          labelId="experiment-name-label"
          onSave={onSave}
        />
      </>
    );
    const { rerender } = render(nameInput('Original name'), { wrapper: ProvidersWrapper });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.paste('Unwanted name');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    expect(onSave).toHaveBeenCalledWith('Unwanted name');

    // While that save is still in flight, restore the original name and leave the field.
    await user.clear(input);
    await user.paste('Original name');
    await user.tab();

    // The in-flight save lands, and the page refetches the experiment with its name.
    rerender(nameInput('Unwanted name'));
    await act(async () => {
      finishFirstSave();
      await Promise.resolve();
    });
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenLastCalledWith('Original name');
    expect(input).toHaveValue('Original name');
  });

  it('keeps a trailing space typed while the rename was saving', async () => {
    const user = userEvent.setup({ advanceTimers: jest.advanceTimersByTime });
    let finishSave: () => void = () => undefined;
    const onSave = jest.fn<ExperimentSettingsNameInputProps['onSave']>().mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          finishSave = resolve;
        }),
    );
    const nameInput = (initialName: string) => (
      <>
        <span id="experiment-name-label">Experiment name</span>
        <ExperimentSettingsNameInput
          initialName={initialName}
          isEditable
          labelId="experiment-name-label"
          onSave={onSave}
        />
      </>
    );
    const { rerender } = render(nameInput('Original name'), { wrapper: ProvidersWrapper });
    const input = screen.getByRole('textbox', { name: 'Experiment name' });

    await user.clear(input);
    await user.paste('Updated name');
    await act(async () => {
      jest.advanceTimersByTime(500);
    });
    expect(onSave).toHaveBeenCalledWith('Updated name');

    // Keep typing while the save is in flight. A trailing space means the same name, so nothing else saves.
    await user.paste(' ');
    expect(input).toHaveValue('Updated name ');

    // The save lands, and the page refetches the experiment with the new name.
    await act(async () => {
      finishSave();
      await Promise.resolve();
    });
    rerender(nameInput('Updated name'));
    await act(async () => Promise.resolve());

    expect(input).toHaveValue('Updated name ');
    expect(onSave).toHaveBeenCalledTimes(1);
  });
});
