import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DesignSystemEventProvider } from '../../DesignSystemEventProvider';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionControlledList } from '../DialogComboboxOptionControlledList';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';

describe('Dialog Combobox - OptionControlledList', () => {
  const eventCallback = jest.fn();

  const Component = ({
    valueHasNoPii,
    multiSelect,
    value,
  }: {
    valueHasNoPii?: boolean;
    multiSelect?: boolean;
    value?: string[];
  }) => (
    <DesignSystemEventProvider callback={eventCallback}>
      <DialogCombobox
        label="example filter"
        componentId="dialog_combobox_test"
        valueHasNoPii={valueHasNoPii}
        multiSelect={multiSelect}
        value={value}
      >
        <DialogComboboxTrigger controlled />
        <DialogComboboxContent>
          <DialogComboboxOptionControlledList options={['one', 'two', 'three']} />
        </DialogComboboxContent>
      </DialogCombobox>
    </DesignSystemEventProvider>
  );

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('does not render outside DialogCombobox', () => {
    const renderOutsideDialogCombobox = () => {
      render(<DialogComboboxOptionControlledList options={[]} />);
    };
    const renderInsideDialogCombobox = () => {
      render(
        <DialogCombobox
          componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptioncontrolledlist.test.tsx_48"
          label="example filter"
        >
          <DialogComboboxOptionControlledList options={[]} />
        </DialogCombobox>,
      );
    };

    jest.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => renderOutsideDialogCombobox()).toThrowError();
    expect(() => renderInsideDialogCombobox()).not.toThrowError();
  });

  it('emits value change events for single select list', async () => {
    render(<Component valueHasNoPii />);
    expect(eventCallback).not.toHaveBeenCalled();

    const combobox = screen.getByRole('combobox', { name: /example filter/ });
    await userEvent.click(combobox);
    expect(eventCallback).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('option', { name: 'one' }));
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'dialog_combobox_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: '["one"]',
    });

    await userEvent.click(combobox);
    await userEvent.click(screen.getByRole('option', { name: 'two' }));
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'dialog_combobox_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: '["two"]',
    });
  });

  it('emits value change events for multiselect list', async () => {
    render(<Component valueHasNoPii multiSelect />);
    expect(eventCallback).not.toHaveBeenCalled();

    const combobox = screen.getByRole('combobox', { name: /example filter/ });
    await userEvent.click(combobox);
    expect(eventCallback).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('option', { name: 'one' }));
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'dialog_combobox_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: '["one"]',
    });

    await userEvent.click(screen.getByRole('option', { name: 'two' }));
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'dialog_combobox_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: '["one","two"]',
    });
  });

  it('emits value change events without value when valueHasNoPii is not set', async () => {
    render(<Component />);
    expect(eventCallback).not.toHaveBeenCalled();

    const combobox = screen.getByRole('combobox', { name: /example filter/ });
    await userEvent.click(combobox);
    expect(eventCallback).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('option', { name: 'one' }));
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'dialog_combobox_test',
      componentType: 'dialog_combobox',
      shouldStartInteraction: false,
      value: undefined,
    });
  });

  it('does not emit a value change event for the default value', async () => {
    render(<Component valueHasNoPii value={['one']} />);
    expect(eventCallback).not.toHaveBeenCalled();
  });
});
