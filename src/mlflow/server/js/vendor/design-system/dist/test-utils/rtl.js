import userEvent from '@testing-library/user-event-14';
import { queryHelpers, waitFor, within } from '@testing-library/react';
import { s as selectClasses, c as createMarkdownTable } from '../common-5b60d682.js';

/**
 * Allows the helpers in this module to be used when the select element is
 * queried _semantically_ (as if it were a native <select> element) - i.e.
 * `ByRole('combobox', { name: '...' })`, rather than by test ID.
 *
 * Also checks if <DesignSystemProvider> was used, because many of the helpers
 * in this module query by class name starting with "du-bois-", which requires
 * the provider.
 */
function getRootElement(element) {
  if (element.getAttribute('role') === 'combobox') {
    element = element.closest(`.${selectClasses.selector}`).parentElement;
  }
  if (element.classList.contains('ant-select')) {
    throw new Error('Component must be wrapped by <DesignSystemProvider>');
  }
  return element;
}
function getOptionsList(select) {
  const body = select.ownerDocument.body;
  const input = within(select).getByRole('combobox');
  const listId = input.getAttribute('aria-owns') || input.getAttribute('aria-controls');
  if (!listId) {
    throw queryHelpers.getElementError('Options input does not control an options list', body);
  }
  const listbox = select.ownerDocument.getElementById(listId);
  if (!(listbox !== null && listbox !== void 0 && listbox.parentElement)) {
    throw queryHelpers.getElementError('Options listbox does not have a parent', body);
  }
  const optionsList = listbox.parentElement.querySelector(`.${selectClasses.list}`);
  if (!optionsList) {
    throw queryHelpers.getElementError('Options list not found', body);
  }
  return optionsList;
}

/**
 * Opens the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already opened or if the menu is unable to be opened.
 */
async function openMenu(select) {
  select = getRootElement(select);
  if (select.classList.contains(selectClasses.open)) {
    throw queryHelpers.getElementError('Select is already open', select);
  }
  const selector = select.querySelector(`.${selectClasses.selector}`);
  if (!selector) {
    throw queryHelpers.getElementError('Selector not found', select);
  }
  await userEvent.click(selector, {
    pointerEventsCheck: 0
  });
  await waitFor(() => {
    if (!select.classList.contains(selectClasses.open)) {
      throw queryHelpers.getElementError('Select did not open', select);
    }
  });
}

/**
 * Closes the dropdown menu for the <Select/> by clicking. Will throw an error if
 * the menu is already closed or if the menu is unable to be closed.
 */
async function closeMenu(select) {
  select = getRootElement(select);
  if (!select.classList.contains(selectClasses.open)) {
    throw queryHelpers.getElementError('Select is already closed', select);
  }
  const selector = select.querySelector(`.${selectClasses.selector}`);
  if (!selector) {
    throw queryHelpers.getElementError('Selector not found', select);
  }
  await userEvent.click(selector, {
    pointerEventsCheck: 0
  });
  await waitFor(() => {
    if (select.classList.contains(selectClasses.open)) {
      throw queryHelpers.getElementError('Select did not close', select);
    }
  });
}

/**
 * Returns a string concatenating the labels for all selected options.
 */
function getLabelText(select) {
  var _selector$textContent, _selector$textContent2;
  select = getRootElement(select);
  const selector = select.querySelector(`.${selectClasses.selector}`);
  if (!selector) {
    throw queryHelpers.getElementError('Selector not found', select);
  }
  // Trim the text to avoid weird whitespace issues non-label elements being added.
  // For example, the input mirror is an empty span with some whitespace that is
  // nested under the selector but does not show up in the label text.
  return (_selector$textContent = (_selector$textContent2 = selector.textContent) === null || _selector$textContent2 === void 0 ? void 0 : _selector$textContent2.trim()) !== null && _selector$textContent !== void 0 ? _selector$textContent : '';
}

/**
 * Removes the `option` by clicking its "X" button. Can only be used with a <Select/>
 * component with `mode="multiple"`. The provided strings must match the option label
 * exactly.
 */
async function removeMultiSelectOption(select, option) {
  select = getRootElement(select);
  const selector = select.querySelector(`.${selectClasses.selector}`);
  if (!selector) {
    throw queryHelpers.getElementError('Selector not found', select);
  }
  const optionItem = within(selector).getByText(option).closest(`.${selectClasses.item}`);
  if (optionItem === null) {
    throw queryHelpers.getElementError(`Option "${option}" not found`, select);
  }
  const removeItem = optionItem.querySelector(`.${selectClasses.removeItem}`);
  if (removeItem === null) {
    throw queryHelpers.getElementError(`Remove button for option "${option}" not found`, optionItem);
  }
  await userEvent.click(removeItem);
}

/**
 * Selects options from the dropdown menu for a <Select/> component with `mode="multiple"`.
 * The provided strings must match the option labels exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
async function multiSelect(select, options) {
  select = getRootElement(select);
  await openMenu(select);
  const optionsList = getOptionsList(select);
  for (let i = 0; i < options.length; i++) {
    const option = options[i];
    const optionItem = within(optionsList).getByText(option);
    await userEvent.click(optionItem, {
      pointerEventsCheck: 0
    });
  }
  // Close the menu to indicate that selection has finished
  await closeMenu(select);
}

/**
 * Selects options from the dropdown menu for a <Select/> component without a
 * mode. The provided string must match an option label exactly. There is a known
 * limitation for lists that are extremely long because AntD virtualizes the
 * options so not all may options may be rendered in the DOM. If this is causing
 * you issues, please let #help-frontend know.
 */
async function singleSelect(select, option) {
  select = getRootElement(select);
  await openMenu(select);
  const optionsList = getOptionsList(select);
  const optionItem = within(optionsList).getByText(option);
  await userEvent.click(optionItem, {
    pointerEventsCheck: 0
  });
  // Menu automatically closes for a single <Select/> (no mode="multiple")
}

/**
 * Clicks on the "Clear" button. In order for this function to work properly,
 * the `allowClear` prop must be set to `true`.
 */
async function clearAll(select) {
  select = getRootElement(select);
  const clearBtn = select.querySelector(`.${selectClasses.clear}`);
  if (!clearBtn) {
    throw queryHelpers.getElementError('Select not clearable', select);
  }
  await userEvent.click(clearBtn);
}

/**
 * Opens the dropdown menu, finds all of the options in the dropdown, closes
 * the menu, and returns a list of the text of each option in order.
 */
async function getAllOptions(select) {
  select = getRootElement(select);
  await openMenu(select);
  const optionsList = getOptionsList(select);
  const options = [];
  optionsList.querySelectorAll(`.${selectClasses.option}`).forEach(option => {
    if (option.textContent === null) {
      throw queryHelpers.getElementError('Option had no text content', option);
    }
    options.push(option.textContent);
  });
  await closeMenu(select);
  return options;
}

/**
 * Creates a new option for a Select with `mode="tags"` by typing it into the input,
 * clicking on the option in the options list, and then closing the menu.
 */
async function createNewOption(select, option) {
  select = getRootElement(select);
  const selectInput = within(select).getByRole('combobox');
  await userEvent.type(selectInput, option);
  const optionsList = getOptionsList(select);
  const optionItem = within(optionsList).getByText(option);
  await userEvent.click(optionItem);
  await closeMenu(select);
}

var selectEvent = /*#__PURE__*/Object.freeze({
  __proto__: null,
  clearAll: clearAll,
  closeMenu: closeMenu,
  createNewOption: createNewOption,
  getAllOptions: getAllOptions,
  getLabelText: getLabelText,
  multiSelect: multiSelect,
  openMenu: openMenu,
  removeMultiSelectOption: removeMultiSelectOption,
  singleSelect: singleSelect
});

function getColumnHeaderIndex(tableElement, columnHeaderName) {
  var _columnHeader$parentE, _columnHeader$parentE2;
  const columnHeader = within(tableElement).getByRole('columnheader', {
    name: columnHeaderName
  });
  const columnHeaderIndex = Array.from((_columnHeader$parentE = (_columnHeader$parentE2 = columnHeader.parentElement) === null || _columnHeader$parentE2 === void 0 ? void 0 : _columnHeader$parentE2.children) !== null && _columnHeader$parentE !== void 0 ? _columnHeader$parentE : []).indexOf(columnHeader);
  return columnHeaderIndex;
}

/**
 * Returns the table row that contains the specified `cellText`. The `cellText`
 * must be in the column with name `columnHeaderName` if it is specified. Otherwise,
 * the `cellText` must be in the first column. Throws an error if either multiple
 * rows or no rows can be found that match the given options. Also throws an error
 * if the column with name `columnHeaderName` cannot be found.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 * @param cellText The cell text that uniquely identifies the row.
 * @param columnHeaderName The name of the column to search the text for. If not provided,
 * the first column will be used.
 */
function getTableRowByCellText(tableElement, cellText) {
  let {
    columnHeaderName
  } = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
  const columnHeaderIndex = columnHeaderName === undefined ? 0 : getColumnHeaderIndex(tableElement, columnHeaderName);
  const matchingRows = within(tableElement).getAllByRole('row')
  // Skip first row (table header)
  .slice(1).filter(row => {
    const cells = within(row).getAllByRole('cell');
    const cell = cells[columnHeaderIndex];
    const cellContainsText = within(cell).queryByText(cellText) !== null;
    return cellContainsText;
  });
  if (matchingRows.length === 0) {
    throw queryHelpers.getElementError(`Unable to find a table row with text "${cellText}" in the column "${columnHeaderName}"`, tableElement);
  }
  if (matchingRows.length > 1) {
    throw queryHelpers.getElementError(`Found multiple table rows with text "${cellText}" in the column "${columnHeaderName}"`, tableElement);
  }
  return matchingRows[0];
}

/**
 * Converts a Du Bois table to a markdown table string. This means that each cell
 * is separated by a pipe (including the edges), the header row is on its own line
 * at the top, each data row is on its own line below, and the header row is separated
 * by a row of dashes from the data rows. This is useful for checking table contents
 * in tests.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 *
 * @example
 * The HTML table:
 * ```jsx
 *   <Table>
 *     <TableRow isHeader>
 *       <TableHeader>Name</TableHeader>
 *       <TableHeader>Fruit</TableHeader>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Alice</TableCell>
 *       <TableCell>Apple</TableCell>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Brady</TableCell>
 *       <TableCell>Banana</TableCell>
 *     </TableRow>
 *   </Table>
 * ```
 *
 * The Markdown table:
 * ```md
 *   | Name | Fruit |
 *   | --- | --- |
 *   | Alice | Apple |
 *   | Brady | Banana |
 * ```
 */
function toMarkdownTable(tableElement) {
  const {
    bodyRows,
    headerRow
  } = getTableRows(tableElement);
  const columns = within(headerRow).getAllByRole('columnheader').map(column => {
    var _column$textContent;
    return (_column$textContent = column.textContent) !== null && _column$textContent !== void 0 ? _column$textContent : '';
  });
  const rows = bodyRows.map(row => within(row).getAllByRole('cell').map(cell => {
    var _cell$textContent;
    return (_cell$textContent = cell.textContent) !== null && _cell$textContent !== void 0 ? _cell$textContent : '';
  }));
  return createMarkdownTable(columns, rows);
}

/**
 * Returns the header row and all body rows (non-header rows) in order. Assumes that the
 * `tableElement` has a single header row (as the first row) and the rest of the rows are
 * body rows.
 *
 * @param tableElement The HTMLElement representing the table to query in. This is likely
 * a `<div role="table">` element, so it can be queried by `screen.getByRole('table')`.
 */
function getTableRows(tableElement) {
  const [firstRow, ...restRows] = within(tableElement).getAllByRole('row');
  return {
    bodyRows: restRows,
    headerRow: firstRow
  };
}

/**
 * Returns the table cell in the specified table row corresponding to the given
 * `columnHeaderName`. This is useful for checking that a row has a particular value
 * for a given column, especially when there are duplicate values in the column.
 *
 * @example
 * The HTML table:
 * ```jsx
 *   <Table>
 *     <TableRow isHeader>
 *       <TableHeader>Name</TableHeader>
 *       <TableHeader>Age</TableHeader>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Alex</TableCell>
 *       <TableCell>25</TableCell>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Brenda</TableCell>
 *       <TableCell>39</TableCell>
 *     </TableRow>
 *     <TableRow>
 *       <TableCell>Carlos</TableCell>
 *       <TableCell>39</TableCell>
 *     </TableRow>
 *   </Table>
 * ```
 *
 * ```js
 * const table = screen.getByRole('table');
 * const result = getTableCellInRow(table, { cellText: 'Carlos' }, 'Age');
 * expect(result.textContent).toEqual('39');
 * ```
 */
function getTableCellInRow(tableElement, row, columnHeaderName) {
  const tableRowElement = getTableRowByCellText(tableElement, row.cellText, {
    columnHeaderName: row.columnHeaderName
  });
  const columnHeaderIndex = getColumnHeaderIndex(tableElement, columnHeaderName);
  const cells = within(tableRowElement).getAllByRole('cell');
  const cell = cells[columnHeaderIndex];
  return cell;
}

/**
 * Opens the dropdown menu by clicking on the dropdown button.
 *
 * @param dropdownButton - The Dropdown Trigger button that opens the menu when clicked.
 */
const openDropdownMenu = async dropdownButton => {
  await userEvent.type(dropdownButton, '{arrowdown}');
};

export { getTableCellInRow, getTableRowByCellText, getTableRows, openDropdownMenu, selectEvent, toMarkdownTable };
//# sourceMappingURL=rtl.js.map
