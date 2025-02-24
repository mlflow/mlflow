import { fireEvent, render, waitFor, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';

import { DesignSystemEventProvider } from '../../DesignSystemEventProvider';
import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';

describe('Dialog Combobox - OptionList', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('does not render outside DialogCombobox', () => {
    const renderOutsideDialogCombobox = () => {
      render(
        <DialogComboboxOptionList>
          <></>
        </DialogComboboxOptionList>,
      );
    };
    const renderInsideDialogCombobox = () => {
      render(
        <DialogCombobox
          componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_28"
          label="example filter"
        >
          <DialogComboboxOptionList>
            <></>
          </DialogComboboxOptionList>
        </DialogCombobox>,
      );
    };

    jest.spyOn(console, 'error').mockImplementation(() => {});
    expect(() => renderOutsideDialogCombobox()).toThrowError();
    expect(() => renderInsideDialogCombobox()).not.toThrowError();
  });

  it('opens on trigger click', async () => {
    const { getByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_43"
        label="example filter"
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            <></>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    fireEvent.click(getByRole('combobox'));
    await waitFor(() => {
      expect(getByRole('listbox')).toBeVisible();
    });
  });

  it('closes on escape', async () => {
    const onOpenChange = jest.fn();

    const { getByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_63"
        label="example filter"
        open={true}
        onOpenChange={onOpenChange}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            <></>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    await waitFor(() => {
      expect(getByRole('combobox')).toBeVisible();
    });
    fireEvent.keyDown(getByRole('combobox'), { key: 'Escape' });

    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it('shows loading state', async () => {
    const options = ['value 1', 'value 2', 'value 3'];
    const { getByRole, getByLabelText, queryByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_84"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList loading={true}>
            {options.map((option) => (
              <DialogComboboxOptionListSelectItem key={option} value={option}>
                {option}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    await waitFor(() => {
      expect(getByRole('combobox')).toBeVisible();
    });

    expect(getByRole('listbox')).toBeVisible();
    expect(getByLabelText('Loading')).toBeVisible();
    expect(queryByRole('option')).not.toBeInTheDocument();
  });

  it('shows loading state with progressive loading', async () => {
    const options = ['value 1', 'value 2', 'value 3'];
    const { getByRole, getByLabelText, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlist.test.tsx_110"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList loading={true} withProgressiveLoading>
            {options.map((option) => (
              <DialogComboboxOptionListSelectItem key={option} value={option}>
                {option}
              </DialogComboboxOptionListSelectItem>
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    await waitFor(() => {
      expect(getByRole('combobox')).toBeVisible();
    });

    expect(getByRole('listbox')).toBeVisible();
    expect(getByLabelText('Loading')).toBeVisible();
    expect(queryAllByRole('option')).toHaveLength(3);
  });

  describe('Analytics Events', () => {
    const eventCallback = jest.fn();

    const SingleSelectComponent = ({
      valueHasNoPii,
      defaultValue,
    }: {
      valueHasNoPii?: boolean;
      defaultValue?: string;
    }) => {
      const [value, setValue] = useState<string>(defaultValue ?? '');
      const options = ['one', 'two', 'three'];

      return (
        <DesignSystemEventProvider callback={eventCallback}>
          <DialogCombobox
            label="example filter"
            componentId="dialog_combobox_test"
            valueHasNoPii={valueHasNoPii}
            value={value ? [value] : []}
          >
            <DialogComboboxTrigger />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {options.map((option) => (
                  <DialogComboboxOptionListSelectItem
                    key={option}
                    value={option}
                    onChange={() => setValue(option)}
                    checked={option === value}
                  />
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </DesignSystemEventProvider>
      );
    };

    const MultiselectComponent = ({
      valueHasNoPii,
      defaultValue,
    }: {
      valueHasNoPii?: boolean;
      defaultValue?: string[];
    }) => {
      const [values, setValues] = useState<string[]>(defaultValue ?? []);
      const options = ['one', 'two', 'three'];

      const handleChange = (option: string) => {
        if (values.includes(option)) {
          setValues(values.filter((value) => value !== option));
        } else {
          setValues([...values, option]);
        }
      };

      return (
        <DesignSystemEventProvider callback={eventCallback}>
          <DialogCombobox
            label="example filter"
            componentId="dialog_combobox_test"
            valueHasNoPii={valueHasNoPii}
            value={values}
            multiSelect
          >
            <DialogComboboxTrigger />
            <DialogComboboxContent>
              <DialogComboboxOptionList>
                {options.map((option) => (
                  <DialogComboboxOptionListCheckboxItem
                    key={option}
                    value={option}
                    onChange={handleChange}
                    checked={values.includes(option)}
                  />
                ))}
              </DialogComboboxOptionList>
            </DialogComboboxContent>
          </DialogCombobox>
        </DesignSystemEventProvider>
      );
    };

    describe('Single Select List', () => {
      it('emits value change events with value', async () => {
        render(<SingleSelectComponent valueHasNoPii />);
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

      it('emits value changes events without value when valueHasNoPii is not set', async () => {
        render(<SingleSelectComponent />);
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
        render(<SingleSelectComponent defaultValue="one" />);
        expect(eventCallback).not.toHaveBeenCalled();
      });
    });

    describe('Multiselect List', () => {
      it('emits value change events with value', async () => {
        render(<MultiselectComponent valueHasNoPii />);
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

      it('emits value changes events without value when valueHasNoPii is not set', async () => {
        render(<MultiselectComponent />);
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
        render(<MultiselectComponent defaultValue={['one', 'two']} />);
        expect(eventCallback).not.toHaveBeenCalled();
      });
    });
  });
});
