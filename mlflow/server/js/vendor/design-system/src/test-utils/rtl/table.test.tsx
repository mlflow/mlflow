import { render, screen, within } from '@testing-library/react';

import { expect } from '@databricks/config-jest';

import { getTableCellInRow, getTableRowByCellText, getTableRows, toMarkdownTable } from './index';
import { DesignSystemProvider, Table, TableCell, TableHeader, TableRow } from '../../design-system';

function renderTable() {
  const data = Array.from({ length: 5 }).map((_, i) => ({
    age: '0',
    id: i,
    name: `Name ${i}`,
  }));
  return render(
    <DesignSystemProvider>
      <Table>
        <TableRow isHeader>
          <TableHeader componentId="codegen_design-system_src_test-utils_rtl_table.test.tsx_18">Name</TableHeader>
          <TableHeader componentId="codegen_design-system_src_test-utils_rtl_table.test.tsx_19">Age</TableHeader>
        </TableRow>
        {data.map((row) => (
          <TableRow key={row.id}>
            <TableCell>{row.name}</TableCell>
            <TableCell>{row.age}</TableCell>
          </TableRow>
        ))}
      </Table>
    </DesignSystemProvider>,
  );
}

describe('getTableRowByCellText', () => {
  it('should return the row that contains the matching cell without specified columnHeaderName', () => {
    renderTable();
    const row = getTableRowByCellText(screen.getByRole('table'), 'Name 2');
    expect(row).toHaveAttribute('role', 'row');
    expect(row).toContainElement(screen.getByText('Name 2'));
  });

  it('should return the row that contains the matching cell with specified columnHeaderName', () => {
    renderTable();
    const row = getTableRowByCellText(screen.getByRole('table'), 'Name 2', { columnHeaderName: 'Name' });
    expect(row).toHaveAttribute('role', 'row');
    expect(row).toContainElement(screen.getByText('Name 2'));
  });

  it('should throw an error when no rows match', () => {
    renderTable();
    expect(() =>
      getTableRowByCellText(screen.getByRole('table'), 'Name 404', { columnHeaderName: 'Name' }),
    ).toThrowError();
  });

  it('should throw an error when more than one row matches', () => {
    renderTable();
    expect(() => getTableRowByCellText(screen.getByRole('table'), '0', { columnHeaderName: 'Age' })).toThrowError();
  });

  it('should throw an error when the column header does not exist', () => {
    renderTable();
    expect(() =>
      getTableRowByCellText(screen.getByRole('table'), 'Name 1', { columnHeaderName: '404' }),
    ).toThrowError();
  });
});

describe('toMarkdownTable', () => {
  it('should return the table in markdown format', () => {
    renderTable();
    const markdownTable = toMarkdownTable(screen.getByRole('table'));
    expect(markdownTable).toEqual(
      `
| Name | Age |
| --- | --- |
| Name 0 | 0 |
| Name 1 | 0 |
| Name 2 | 0 |
| Name 3 | 0 |
| Name 4 | 0 |
    `.trim(),
    );
  });
});

describe('getTableRows', () => {
  it('should return the header row and all body rows in order', () => {
    renderTable();
    const result = getTableRows(screen.getByRole('table'));
    expect(result.headerRow).toBeDefined();
    expect(result.bodyRows).toHaveLength(5);
    result.bodyRows.forEach((row, index) => {
      expect(within(row).getByText(`Name ${index}`)).toBeInTheDocument();
    });
  });
});

describe('getTableCellInRow', () => {
  it('should return the cell for the appropriate column in the specified row', () => {
    renderTable();
    const table = screen.getByRole('table');
    expect(getTableCellInRow(table, { cellText: 'Name 0' }, 'Age').textContent).toEqual('0');
  });
});
