import { fireEvent, render } from '@testing-library/react';

import { TableCell, TableHeader, TableRow, TableRowSelectCell, Table, TableFilterInput } from './index';

const Example = (): JSX.Element => {
  return (
    <div>
      <Table>
        <TableRow isHeader>
          <TableHeader componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_10">
            Name
          </TableHeader>
          <TableHeader componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_11">
            Age
          </TableHeader>
          <TableHeader componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_12">
            Address
          </TableHeader>
        </TableRow>

        <TableRow>
          <TableCell>George</TableCell>
          <TableCell>30</TableCell>
          <TableCell>1313 Mockingbird Lane</TableCell>
        </TableRow>
        <TableRow>
          <TableCell>Joe</TableCell>
          <TableCell>31</TableCell>
          <TableCell>1313 Mockingbird Lane</TableCell>
        </TableRow>
      </Table>
    </div>
  );
};

describe('TableUI', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('Renders correctly', () => {
    const { queryAllByRole } = render(<Example />);

    // Validate that roles are set correctly.
    expect(queryAllByRole('table')).toHaveLength(1);
    expect(queryAllByRole('row')).toHaveLength(3);
    expect(queryAllByRole('columnheader')).toHaveLength(3);
    expect(queryAllByRole('cell')).toHaveLength(6);
  });

  describe('TableHeader', () => {
    it('throws an error if you try to use TableHeader without TableRow + `isHeader` prop', () => {
      const renderBadTable = () => {
        render(
          <Table>
            <TableRow>
              <TableHeader componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_51">
                Name
              </TableHeader>
            </TableRow>
          </Table>,
        );
      };

      const renderCorrectTable = () => {
        render(
          <Table>
            <TableRow isHeader>
              <TableHeader componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_61">
                Name
              </TableHeader>
            </TableRow>
          </Table>,
        );
      };

      // Suppress console.error output.
      jest.spyOn(console, 'error').mockImplementation(() => {});
      expect(() => renderBadTable()).toThrowError();
      expect(() => renderCorrectTable()).not.toThrowError();
    });
  });

  describe('TableRowSelectCell', () => {
    it('throws an error if you try to use `TableRowSelectCell` without providing `someRowsSelected` to `Table`', () => {
      const renderBadTable = () => {
        render(
          <Table>
            <TableRow>
              <TableRowSelectCell
                componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_80"
                checked
              />
            </TableRow>
          </Table>,
        );
      };

      const renderCorrectTable = () => {
        render(
          <Table someRowsSelected>
            <TableRow isHeader>
              <TableRowSelectCell
                componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_90"
                checked
              />
            </TableRow>
          </Table>,
        );
      };

      // Suppress console.error output.
      jest.spyOn(console, 'error').mockImplementation(() => {});
      expect(() => renderBadTable()).toThrowError();
      expect(() => renderCorrectTable()).not.toThrowError();
    });

    it('throws an error if you try to set `TableRowSelectCell` to `indeterminate` outside of a header row', () => {
      const renderBadTable = () => {
        render(
          <Table someRowsSelected>
            <TableRow>
              <TableRowSelectCell
                componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_107"
                indeterminate
              />
            </TableRow>
          </Table>,
        );
      };

      const renderCorrectTable = () => {
        render(
          <Table someRowsSelected>
            <TableRow isHeader>
              <TableRowSelectCell
                componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_117"
                indeterminate
              />
            </TableRow>
          </Table>,
        );
      };

      // Suppress console.error output.
      jest.spyOn(console, 'error').mockImplementation(() => {});
      expect(() => renderBadTable()).toThrowError();
      expect(() => renderCorrectTable()).not.toThrowError();
    });

    it('should call onChange when TableFilterInput is called', () => {
      const onChange = jest.fn();

      const { getByTestId } = render(
        <TableFilterInput
          componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_133"
          data-testid="filter-input"
          onChange={onChange}
        />,
      );

      const input = getByTestId('filter-input');
      fireEvent.change(input, { target: { value: 'test' } });

      expect(onChange).toHaveBeenCalledTimes(1);
    });

    it('should call onClear when clear icon is pressed in TableFilterInput', async () => {
      const onClear = jest.fn();

      const { getByRole } = render(
        <TableFilterInput
          componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_143"
          onClear={onClear}
          value="val"
        />,
      );

      const clearIcon = getByRole('button');
      fireEvent.click(clearIcon);

      expect(onClear).toHaveBeenCalledTimes(1);
    });

    it('should call onChange when onClear is not defined and clear icon is clicked', async () => {
      const onChange = jest.fn();

      const { getByRole } = render(
        <TableFilterInput
          componentId="codegen_design-system_src_design-system_tableui_tableui.test.tsx_154"
          onChange={onChange}
          value="val"
        />,
      );

      const clearIcon = getByRole('button');
      fireEvent.click(clearIcon);

      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });
});
