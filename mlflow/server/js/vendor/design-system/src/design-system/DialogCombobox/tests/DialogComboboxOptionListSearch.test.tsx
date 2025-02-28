import { render, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { DialogCombobox } from '../DialogCombobox';
import { DialogComboboxContent } from '../DialogComboboxContent';
import { DialogComboboxOptionList } from '../DialogComboboxOptionList';
import { DialogComboboxOptionListCheckboxItem } from '../DialogComboboxOptionListCheckboxItem';
import { DialogComboboxOptionListSearch } from '../DialogComboboxOptionListSearch';
import { DialogComboboxOptionListSelectItem } from '../DialogComboboxOptionListSelectItem';
import { DialogComboboxTrigger } from '../DialogComboboxTrigger';

const items: any[] = [
  {
    key: '1',
    value: 'Alpha',
  },
  {
    key: '2',
    value: 'Beta',
  },
  {
    key: '3',
    value: 'Charlie',
  },
  {
    key: '4',
    value: 'Delta',
  },
];

describe('Dialog Combobox - Search', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('correctly filters DialogComboboxOptionListSelectItem children', () => {
    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistsearch.test.tsx_38"
        label="Owner"
        open={true}
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent minWidth={250}>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch>
              {items.map((item) => (
                <DialogComboboxOptionListSelectItem key={item.key} value={item.value} />
              ))}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    const input = getByRole('searchbox');
    userEvent.type(input, 'A');
    // eslint-disable-next-line testing-library/await-async-utils -- FEINF-3005
    waitFor(() => {
      expect(queryAllByRole('option')).toHaveLength(1);
    });
  });

  it('correctly filters DialogComboboxOptionListCheckboxItem children', () => {
    const { getByRole, queryAllByRole } = render(
      <DialogCombobox
        componentId="codegen_design-system_src_design-system_dialogcombobox_tests_dialogcomboboxoptionlistsearch.test.tsx_62"
        label="Owner"
        open={true}
        multiSelect
      >
        <DialogComboboxTrigger />
        <DialogComboboxContent minWidth={250}>
          <DialogComboboxOptionList>
            <DialogComboboxOptionListSearch>
              {items.map((item) => (
                <DialogComboboxOptionListCheckboxItem key={item.key} value={item.value} />
              ))}
            </DialogComboboxOptionListSearch>
          </DialogComboboxOptionList>
        </DialogComboboxContent>
      </DialogCombobox>,
    );

    const input = getByRole('searchbox');
    userEvent.type(input, 'A');
    // eslint-disable-next-line testing-library/await-async-utils -- FEINF-3005
    waitFor(() => {
      expect(queryAllByRole('option')).toHaveLength(1);
    });
  });
});
