import { fireEvent, render, waitFor } from '@testing-library/react';

import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';

describe('Dialog Combobox - SelectItem', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders', async () => {
    const options = ['value 1', 'value 2', 'value 3'];

    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_18"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
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

    expect(getByRole('combobox')).toBeVisible();
    expect(queryAllByRole('option')).toHaveLength(3);
  });

  it('does not render outside of DialogComboboxOptionList', async () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() =>
      render(<DialogComboboxOptionListSelectItem value="value 1">value 1</DialogComboboxOptionListSelectItem>),
    ).toThrowError('`DialogComboboxOptionListSelectItem` must be used within `DialogComboboxOptionList`');
  });

  it('renders with selected state', async () => {
    const options = ['value 1', 'value 2', 'value 3'];

    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_52"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {options.map((option) => (
              <DialogComboboxOptionListSelectItem key={option} value={option} checked={option === 'value 2'}>
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

    expect(getByRole('combobox')).toBeVisible();
    expect(queryAllByRole('option')).toHaveLength(3);
    expect(queryAllByRole('option')[1]).toHaveAttribute('aria-selected', 'true');
  });

  it('propagates changes', async () => {
    const options = ['value 1', 'value 2', 'value 3'];
    const onChange = jest.fn();

    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_80"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {options.map((option) => (
              <DialogComboboxOptionListSelectItem key={option} value={option} onChange={onChange}>
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

    expect(getByRole('combobox')).toBeVisible();
    expect(queryAllByRole('option')).toHaveLength(3);

    fireEvent.click(queryAllByRole('option')[1]);

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith('value 2', expect.anything());
    });
  });

  it('navigates with keyboard', async () => {
    const options = ['value 1', 'value 2', 'value 3'];

    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_112"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
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

    expect(getByRole('combobox')).toBeVisible();
    expect(queryAllByRole('option')).toHaveLength(3);

    fireEvent.keyDown(getByRole('combobox'), { key: 'ArrowDown' });

    const optionElements = queryAllByRole('option');

    await waitFor(() => {
      expect(optionElements[0]).toHaveFocus();
    });

    fireEvent.keyDown(optionElements[0], { key: 'ArrowDown' });

    await waitFor(() => {
      expect(optionElements[1]).toHaveFocus();
    });

    fireEvent.keyDown(optionElements[1], { key: 'ArrowUp' });

    await waitFor(() => {
      expect(optionElements[0]).toHaveFocus();
    });
  });

  it('should not trigger `onChange` when disabled and help icon is clicked', async () => {
    const onChange = jest.fn();

    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistselectitem.test.tsx_158"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSelectItem key="value 1" value="value 1" onChange={onChange} disabled>
              value 1
            </DialogComboboxOptionListSelectItem>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    await waitFor(() => {
      expect(getByRole('combobox')).toBeVisible();
    });

    expect(getByRole('combobox')).toBeVisible();
    expect(queryAllByRole('option')).toHaveLength(1);

    fireEvent.click(queryAllByRole('option')[0]);

    expect(onChange).not.toHaveBeenCalled();
  });
});
