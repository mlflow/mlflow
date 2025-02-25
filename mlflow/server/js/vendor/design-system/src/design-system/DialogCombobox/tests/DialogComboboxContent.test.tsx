import { render } from '@testing-library/react';

import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';

describe('Dialog Combobox - Content', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders', () => {
    const options = ['value 1', 'value 2', 'value 3'];
    const { getByLabelText, queryByLabelText } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxcontent.test.tsx_17"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent>
          <DialogComboboxOptionList>
            {options.map((option, key) => (
              <DialogComboboxOptionListSelectItem key={key} value={option} />
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    const content = getByLabelText('example filter options');
    expect(content).toBeVisible();
    expect(content.getAttribute('aria-busy')).toBeFalsy();
    expect(queryByLabelText('Loading')).toBeFalsy();
  });

  it('renders loading', () => {
    const options = ['value 1', 'value 2', 'value 3'];
    const { getByLabelText, queryByText } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxcontent.test.tsx_38"
        label="example filter"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent loading>
          <DialogComboboxOptionList>
            {options.map((option, key) => (
              <DialogComboboxOptionListSelectItem key={key} value={option} />
            ))}
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    const content = getByLabelText('example filter options');
    expect(content).toBeVisible();
    expect(content.getAttribute('aria-busy')).toBeTruthy();
    expect(queryByText('Loading')).toBeTruthy();
  });

  it("doesn't render outside DialogCombobox", () => {
    jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => render(<DialogComboboxContent aria-label="Buttons container" />)).toThrowError(
      '`DialogComboboxContent` must be used within `DialogCombobox`',
    );
  });
});
