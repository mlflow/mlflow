import { fireEvent, render, waitFor } from '@testing-library/react';

import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';

describe('Dialog Combobox', () => {
  it('should render', () => {
    const { getByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcombobox.test.tsx_13"
        label="render test"
      >
        <DialogComboboxTrigger />
      </DialogCombobox>,
    );
    expect(getByRole('combobox')).toBeVisible();
  });

  it('should render once on value change from within', async () => {
    const onChange = jest.fn();

    const { getByRole, getByText } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcombobox.test.tsx_24"
        label="render test"
        value={[]}
        open
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSelectItem value="one" onChange={onChange}>
              one
            </DialogComboboxOptionListSelectItem>
            <DialogComboboxOptionListSelectItem value="two" onChange={onChange}>
              two
            </DialogComboboxOptionListSelectItem>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );
    expect(getByRole('combobox')).toBeVisible();

    fireEvent.click(getByText('one'));
    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith('one', expect.anything());
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });

  it('should render once on value change from within multiselect', async () => {
    const onChange = jest.fn();
    const { queryAllByRole, getByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcombobox.test.tsx_50"
        label="render test"
        multiSelect
        value={[]}
        open
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListCheckboxItem value="one" onChange={onChange}>
              one
            </DialogComboboxOptionListCheckboxItem>
            <DialogComboboxOptionListCheckboxItem value="two" onChange={onChange}>
              two
            </DialogComboboxOptionListCheckboxItem>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    await waitFor(() => expect(getByRole('listbox')).toBeVisible());

    const optionsElements = queryAllByRole('option');
    expect(optionsElements).toHaveLength(2);

    fireEvent.click(optionsElements[0]);
    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith('one', expect.anything());
      expect(onChange).toHaveBeenCalledTimes(1);
    });

    fireEvent.click(optionsElements[1]);
    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith('two', expect.anything());
      expect(onChange).toHaveBeenCalledTimes(2);
    });
  });
});
