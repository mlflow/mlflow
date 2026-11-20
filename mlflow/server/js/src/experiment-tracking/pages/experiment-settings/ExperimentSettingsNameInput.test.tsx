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
    expect(input).toHaveAccessibleDescription('Unable to save the experiment name: rename failed Retry');
    await user.click(screen.getByRole('button', { name: 'Retry' }));
    await act(async () => Promise.resolve());

    expect(onSave).toHaveBeenCalledTimes(2);
    expect(onSave).toHaveBeenLastCalledWith('Retained draft');
    expect(screen.queryByText(/Unable to save the experiment name/)).not.toBeInTheDocument();
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

  it('shows the server reason when a rename fails', async () => {
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
    expect(
      await screen.findByText("Unable to save the experiment name: Experiment 'Taken name' already exists."),
    ).toBeVisible();
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
});
