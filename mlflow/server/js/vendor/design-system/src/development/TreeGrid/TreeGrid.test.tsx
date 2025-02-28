import {
  DesignSystemProvider,
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  FileIcon,
  FolderIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';

import type {
  TreeGridProps,
  TreeGridRenderCellArgs,
  TreeGridRenderRowArgs,
  TreeGridRow,
  TreeGridColumn,
  TreeGridState,
} from '.';
import { TreeGrid } from '.';

const sampleData: TreeGridRow[] = [
  {
    id: '1',
    name: 'Root 1',
    type: 'Folder',
    buttonId: '1',
    children: [
      { id: '1-1', name: 'Child 1-1', type: 'File', buttonId: '1-1', widgetCount: 3 },
      {
        id: '1-2',
        name: 'Child 1-2',
        type: 'Folder',
        buttonId: '1-2',
        children: [{ id: '1-2-1', name: 'Grandchild 1-2-1', type: 'File', buttonId: '1-2-1' }],
      },
      { id: '1-3', name: 'Child 1-3', type: 'File', widgetCount: 5 },
    ],
  },
  {
    id: '2',
    name: 'Root 2',
    type: 'Folder',
    buttonId: '2',
    children: [
      { id: '2-1', name: 'Child 2-1', type: 'File', widgetCount: 1 },
      { id: '2-2', name: 'Child 2-2', type: 'File', buttonId: '2-2', widgetCount: 4 },
    ],
  },
];

const columns: TreeGridColumn[] = [
  { id: 'name', header: 'Name', isRowHeader: true },
  { id: 'type', header: 'Type' },
  { id: 'buttonId', header: 'Actions', contentFocusable: true },
  { id: 'widgetCount', header: 'Widget Count' },
];

const DefaultTreeGridExample: React.FC<Partial<TreeGridProps>> = (props) => {
  const { theme } = useDesignSystemTheme();

  const renderCell = ({
    row,
    column,
    rowDepth,
    rowIsKeyboardActive,
    rowIsExpanded,
    toggleRowExpanded,
    cellProps,
  }: TreeGridRenderCellArgs) => {
    const content = (() => {
      switch (column.id) {
        case 'name':
          return (
            <span
              style={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs, height: 24 }}
              data-testid="name"
            >
              <span style={{ marginLeft: `${(rowDepth || 0) * theme.spacing.sm}px` }} />
              {row.children && (
                <span onClick={() => toggleRowExpanded(row.id)}>
                  {rowIsExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
                </span>
              )}
              <span>{row.children ? <FolderIcon /> : <FileIcon />}</span>
              <span>{row['name']}</span>
            </span>
          );
        case 'type':
          return row['type'];
        case 'buttonId':
          return (
            row['buttonId'] && (
              <Button
                size="small"
                componentId={`button-${row['buttonId']}`}
                aria-label={`button-${row['buttonId']}`}
                tabIndex={rowIsKeyboardActive ? 0 : -1}
              >
                {row['buttonId']}
              </Button>
            )
          );
        case 'widgetCount':
          return row['widgetCount'] ? <span data-testid="widget-count">{row['widgetCount']}</span> : null;
        default:
          return row[column.id as keyof TreeGridRow];
      }
    })();

    return <td {...cellProps}>{content}</td>;
  };

  return (
    <DesignSystemProvider>
      <TreeGrid includeHeader data={sampleData} columns={columns} renderCell={renderCell} {...props} />
    </DesignSystemProvider>
  );
};

const fireKeyOnActiveElement = (key: string) => {
  fireEvent.keyDown(document.activeElement!, { key });
};

describe('TreeGrid', () => {
  it('should render TreeGrid with correct structure', () => {
    render(<DefaultTreeGridExample />);

    const treegrid = screen.getByRole('treegrid');
    expect(treegrid).toBeInTheDocument();

    const rows = screen.getAllByRole('row');
    expect(rows).toHaveLength(3); // 2 root rows + header row

    const headerCells = screen.getAllByRole('columnheader');
    expect(headerCells).toHaveLength(4);
    expect(headerCells[0]).toHaveTextContent('Name');
    expect(headerCells[1]).toHaveTextContent('Type');
    expect(headerCells[2]).toHaveTextContent('Actions');
    expect(headerCells[3]).toHaveTextContent('Widget Count');

    const actionButtons = screen.getAllByRole('button', { name: /button/i });
    expect(actionButtons).toHaveLength(2); // One for each root row
  });

  it('should have correct ARIA attributes on rows and update them when expanded', async () => {
    render(<DefaultTreeGridExample />);

    const rows = screen.getAllByRole('row');

    expect(rows[1]).toHaveAttribute('aria-level', '1');
    expect(rows[1]).toHaveAttribute('aria-expanded', 'false');
    expect(rows[2]).toHaveAttribute('aria-level', '1');
    expect(rows[2]).toHaveAttribute('aria-expanded', 'false');

    const firstRow = rows[1];
    firstRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      const updatedRows = screen.getAllByRole('row');
      expect(updatedRows[1]).toHaveAttribute('aria-expanded', 'true');
      expect(updatedRows[2]).toHaveAttribute('aria-level', '2');
      expect(updatedRows[2]).not.toHaveAttribute('aria-expanded');
      expect(updatedRows[3]).toHaveAttribute('aria-level', '2');
      expect(updatedRows[3]).toHaveAttribute('aria-expanded', 'false');
      expect(updatedRows[4]).toHaveAttribute('aria-level', '2');
      expect(updatedRows[4]).not.toHaveAttribute('aria-expanded');
      expect(updatedRows[5]).toHaveAttribute('aria-level', '1');
      expect(updatedRows[5]).toHaveAttribute('aria-expanded', 'false');
    });
  });

  it('should expand/collapse rows on Right/Left Arrow keys', async () => {
    render(<DefaultTreeGridExample />);

    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();

    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
      const childRow = screen.getByRole('row', { name: /Child 1-1/i });
      expect(childRow).toBeInTheDocument();
    });

    fireKeyOnActiveElement('ArrowLeft');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'false');
      const childRow = screen.queryByRole('row', { name: /Child 1-1/i });
      expect(childRow).not.toBeInTheDocument();
    });
  });

  it('should focus on the first cell when the row is focused and the RIGHT key is pressed (and the row is expandable and expanded)', async () => {
    render(<DefaultTreeGridExample />);

    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();

    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
      expect(rootRow).toHaveFocus();
    });

    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      const firstCell = screen.getAllByTestId('name')[0].parentElement;
      expect(firstCell).toHaveFocus();
    });
  });

  it('For a non-expandable row, if you hit right, the first cell gains focus', async () => {
    render(<DefaultTreeGridExample />);

    // Expand the root row to access its children
    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
    });

    fireKeyOnActiveElement('ArrowDown');

    const leafRow = screen.getByRole('row', { name: /Child 1-1/i });

    expect(leafRow).toHaveFocus();

    fireKeyOnActiveElement('ArrowRight');

    const firstCell = leafRow.querySelector('[role="rowheader"]');
    expect(firstCell).toHaveFocus();
  });

  it('should return focus to the row if on the leftmost cell and you hit left.', async () => {
    render(<DefaultTreeGridExample />);

    // Expand the root row to access its children
    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
    });

    // Move focus to the first child row
    fireKeyOnActiveElement('ArrowDown');

    const childRow = screen.getByRole('row', { name: /Child 1-1/i });
    expect(childRow).toHaveFocus();

    // Move focus to the first cell of the child row
    fireKeyOnActiveElement('ArrowRight');

    const firstCell = childRow.querySelector('[role="rowheader"]');
    expect(firstCell).toHaveFocus();

    // Press left arrow key on the first cell
    fireKeyOnActiveElement('ArrowLeft');

    // Check if focus returned to the row
    await waitFor(() => {
      expect(childRow).toHaveFocus();
    });
  });

  it('should move focus to the parent row when left arrow is pressed on a child row', async () => {
    render(<DefaultTreeGridExample />);

    // Expand the root row to access its children
    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
    });

    // Move focus to the first child row
    fireKeyOnActiveElement('ArrowDown');

    const childRow = screen.getByRole('row', { name: /Child 1-1/i });
    expect(childRow).toHaveFocus();

    // Press left arrow key on the child row
    fireKeyOnActiveElement('ArrowLeft');

    // Check if focus moved to the parent row
    expect(rootRow).toHaveFocus();

    // Move focus to the third child row
    const moveDown = (times: number) => {
      for (let i = 0; i < times; i++) {
        fireKeyOnActiveElement('ArrowDown');
      }
    };

    moveDown(3);

    const thirdChildRow = screen.getByRole('row', { name: /Child 1-3/i });
    expect(thirdChildRow).toHaveFocus();

    // Press left arrow key on the third child row
    fireKeyOnActiveElement('ArrowLeft');

    // Check if focus moved to the parent row
    expect(rootRow).toHaveFocus();
  });

  it('should navigate between rows using arrow keys', async () => {
    render(<DefaultTreeGridExample />);

    const rows = screen.getAllByRole('row');
    rows[1].focus();

    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      expect(rows[2]).toHaveFocus();
    });

    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      expect(rows[1]).toHaveFocus();
    });
  });

  it('should navigate horizontally between cells using arrow keys', async () => {
    render(<DefaultTreeGridExample />);

    const firstCell = screen.getByRole('rowheader', { name: /Root 1/i });
    const row = firstCell.closest('tr');
    const secondCell = row?.querySelector('[role="gridcell"]');

    firstCell.focus();

    fireKeyOnActiveElement('ArrowRight');
    await waitFor(() => {
      expect(secondCell).toHaveFocus();
    });

    fireKeyOnActiveElement('ArrowLeft');
    await waitFor(() => {
      expect(firstCell).toHaveFocus();
    });
  });

  it('should navigate vertically between cells using arrow keys', async () => {
    render(<DefaultTreeGridExample />);

    const firstRowHeader = screen.getByRole('rowheader', { name: /Root 1/i });
    const secondRowHeader = screen.getByRole('rowheader', { name: /Root 2/i });

    const firstRowCell = firstRowHeader.nextElementSibling as HTMLElement;
    const secondRowCell = secondRowHeader.nextElementSibling as HTMLElement;

    // Start with focus on the first row's first cell
    firstRowHeader.focus();

    // Navigate down
    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      expect(secondRowHeader).toHaveFocus();
    });

    // Navigate back up
    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      expect(firstRowHeader).toHaveFocus();
    });

    // Test navigation in the second column
    firstRowCell.focus();

    // Navigate down in the second column
    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      expect(secondRowCell).toHaveFocus();
    });

    // Navigate back up in the second column
    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      expect(firstRowCell).toHaveFocus();
    });
  });

  it('should navigate in a square pattern: down, right, up, left', async () => {
    render(<DefaultTreeGridExample />);

    const firstRowHeader = screen.getByRole('rowheader', { name: /Root 1/i });
    const secondRowHeader = screen.getByRole('rowheader', { name: /Root 2/i });

    const firstRowCell = firstRowHeader.nextElementSibling as HTMLElement;
    const secondRowCell = secondRowHeader.nextElementSibling as HTMLElement;

    // Start with focus on the first row's first cell (top-left)
    firstRowHeader.focus();

    // 1. Navigate down
    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      expect(secondRowHeader).toHaveFocus();
    });

    // 2. Navigate right
    fireKeyOnActiveElement('ArrowRight');
    await waitFor(() => {
      expect(secondRowCell).toHaveFocus();
    });

    // 3. Navigate up
    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      expect(firstRowCell).toHaveFocus();
    });

    // 4. Navigate left
    fireKeyOnActiveElement('ArrowLeft');
    await waitFor(() => {
      expect(firstRowHeader).toHaveFocus();
    });
  });

  it('should navigate correctly between buttons in different rows', async () => {
    render(<DefaultTreeGridExample />);

    // Expand the first root row
    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    // Expand the "Child 1-2" row
    await waitFor(() => {
      const childRow = screen.getByRole('row', { name: /Child 1-2/i });
      childRow.focus();
      fireKeyOnActiveElement('ArrowRight');
    });

    // Find and focus the button for 1-2-1
    await waitFor(() => {
      const button = screen.getByRole('button', { name: 'button-1-2-1' });
      button.focus();
      expect(button).toHaveFocus();
    });

    // Navigate up to button 1-2
    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      const button = screen.getByRole('button', { name: 'button-1-2' });
      expect(button).toHaveFocus();
    });

    // Navigate down
    fireKeyOnActiveElement('ArrowDown');
    fireKeyOnActiveElement('ArrowDown');

    // Check that we've landed on button 2 (skipping 1-3, which doesn't have a button)
    await waitFor(() => {
      const button = screen.getByRole('button', { name: 'button-2' });
      expect(button).toHaveFocus();
    });
  });

  it('should apply correct tabindex to regular gridcells', () => {
    render(<DefaultTreeGridExample />);

    const firstRow = screen.getByRole('row', { name: /Root 1/i });
    const firstRowCell = firstRow.querySelector('td');
    const secondRow = screen.getByRole('row', { name: /Root 2/i });
    const secondRowCell = secondRow.querySelector('td');

    expect(firstRowCell).toHaveAttribute('tabindex', '0');
    expect(secondRowCell).toHaveAttribute('tabindex', '-1');
  });

  it('should apply correct tabindex to button cells and buttons', () => {
    render(<DefaultTreeGridExample />);

    const firstRow = screen.getByRole('row', { name: /Root 1/i });
    const firstRowButton = firstRow.querySelector('button');
    const firstRowButtonCell = firstRowButton?.closest('td');
    const secondRow = screen.getByRole('row', { name: /Root 2/i });
    const secondRowButton = secondRow.querySelector('button');
    const secondRowButtonCell = secondRowButton?.closest('td');

    firstRow.focus();

    expect(firstRowButton).toHaveAttribute('tabindex', '0');
    expect(firstRowButtonCell).not.toHaveAttribute('tabindex');
    expect(secondRowButton).toHaveAttribute('tabindex', '-1');
    expect(secondRowButtonCell).not.toHaveAttribute('tabindex');
  });

  it('should handle custom cell rendering', () => {
    const customRenderCell = ({ row, column, cellProps }: TreeGridRenderCellArgs) => {
      if (column.id === 'type') {
        return (
          <td {...cellProps} data-testid="custom-cell">
            {row[column.id]}
          </td>
        );
      }
      return <td {...cellProps}>{row[column.id]}</td>;
    };

    render(<DefaultTreeGridExample renderCell={customRenderCell} />);

    const customCells = screen.getAllByTestId('custom-cell');
    expect(customCells).toHaveLength(2);
    expect(customCells[0]).toHaveTextContent('Folder');
  });

  it('should handle onCellSelect callback', async () => {
    const onCellSelect = jest.fn();
    render(<DefaultTreeGridExample onCellKeyboardSelect={onCellSelect} />);

    const firstCell = screen.getByRole('rowheader', { name: /Root 1/i });
    firstCell.focus();

    fireKeyOnActiveElement('Enter');
    await waitFor(() => {
      expect(onCellSelect).toHaveBeenCalledWith('1', 'name');
    });
  });

  it('should handle onRowSelect callback', async () => {
    const onRowSelect = jest.fn();
    render(<DefaultTreeGridExample onRowKeyboardSelect={onRowSelect} />);

    const firstRow = screen.getByRole('row', { name: /Root 1/i });
    firstRow.focus();

    fireKeyOnActiveElement('Enter');
    await waitFor(() => {
      expect(onRowSelect).toHaveBeenCalledWith('1');
    });
  });

  it('should apply custom row rendering', () => {
    const customClass = 'custom-row-class';
    const customStyle = { backgroundColor: 'red' };

    const renderRow = jest.fn().mockImplementation(({ row, rowProps, children }: TreeGridRenderRowArgs) => (
      <tr
        {...rowProps}
        className={`${rowProps.className || ''} ${customClass}`.trim()}
        style={{ ...rowProps.style, ...customStyle }}
        data-testid={`row-${row.id}`}
      >
        {children}
      </tr>
    ));

    const renderCell = jest
      .fn()
      .mockImplementation(({ row, column, cellProps }) => <td {...cellProps}>{row[column.id]}</td>);

    const props: TreeGridProps = {
      data: sampleData,
      columns,
      renderCell,
      renderRow,
      includeHeader: false,
    };

    render(<TreeGrid {...props} />);

    const rows = screen.getAllByRole('row');
    rows.forEach((row, index) => {
      expect(row).toHaveClass(customClass);
      expect(row).toHaveStyle('background-color: red');
      expect(row).toHaveAttribute('data-testid', `row-${sampleData[index].id}`);
    });

    expect(renderRow).toHaveBeenCalledTimes(sampleData.length);
  });

  it('should navigate vertically between interactive cells using arrow keys, skipping empty cells', async () => {
    render(<DefaultTreeGridExample />);

    // Expand the first root row
    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
    });

    // Focus on the first button
    const firstButton = screen.getByRole('button', { name: 'button-1' });
    firstButton.focus();

    // Navigate down
    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      const secondButton = screen.getByRole('button', { name: 'button-1-1' });
      expect(secondButton).toHaveFocus();
    });

    // Navigate down again (should skip the row without a button)
    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      const thirdButton = screen.getByRole('button', { name: 'button-1-2' });
      expect(thirdButton).toHaveFocus();
    });

    // Navigate up
    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      const secondButton = screen.getByRole('button', { name: 'button-1-1' });
      expect(secondButton).toHaveFocus();
    });
  });

  it('should navigate vertically between non-interactive cells using arrow keys, skipping empty cells', async () => {
    render(<DefaultTreeGridExample />);

    // Expand the first root row
    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();
    fireKeyOnActiveElement('ArrowRight');

    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
    });

    // Focus on the first 'widget count' cell
    const firstWidgetCountCell = screen.getAllByTestId('widget-count')[0].closest('td');
    firstWidgetCountCell?.focus();

    // Navigate down (should skip the row without a 'widget count' value)
    fireKeyOnActiveElement('ArrowDown');
    await waitFor(() => {
      const secondWidgetCountCell = screen.getAllByTestId('widget-count')[1].closest('td');
      expect(document.activeElement).toBe(secondWidgetCountCell);
    });

    // Navigate up
    fireKeyOnActiveElement('ArrowUp');
    await waitFor(() => {
      expect(document.activeElement).toBe(firstWidgetCountCell);
    });
  });

  it('should collapse an expanded row when focused on the row and hitting LEFT', async () => {
    render(<DefaultTreeGridExample />);

    const rootRow = screen.getByRole('row', { name: /Root 1/i });
    rootRow.focus();

    // Expand the row
    fireKeyOnActiveElement('ArrowRight');
    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'true');
    });

    // Collapse the row
    fireKeyOnActiveElement('ArrowLeft');
    await waitFor(() => {
      expect(rootRow).toHaveAttribute('aria-expanded', 'false');
    });
  });

  it('should apply initial expanded state', async () => {
    const initialState: TreeGridState = {
      expandedRows: { '1': true, '1-2': true },
    };

    render(<DefaultTreeGridExample initialState={initialState} />);

    await waitFor(() => {
      const root1Row = screen.getByRole('row', { name: /^Root 1/ });
      expect(root1Row).toHaveAttribute('aria-expanded', 'true');

      const child12Row = screen.getByRole('row', { name: /^Child 1-2/ });
      expect(child12Row).toHaveAttribute('aria-expanded', 'true');

      const child11Row = screen.getByRole('row', { name: /^Child 1-1/ });
      expect(child11Row).toBeInTheDocument();

      const grandchild121Row = screen.getByRole('row', { name: /^Grandchild 1-2-1/ });
      expect(grandchild121Row).toBeInTheDocument();
    });
  });
});
