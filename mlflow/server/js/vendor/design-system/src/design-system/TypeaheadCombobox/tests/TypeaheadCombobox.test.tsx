import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import {
  books,
  getFilteredBooks,
  matcher,
  renderSingleSelectTypeahead,
  renderMultiSelectTypeahead,
  openMenuWithButton,
  closeMenu,
  selectItemByText,
  type Book,
  renderSingleSelectTypeaheadWithLabel,
} from './testFixtures';
import {
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxMultiSelectInput,
  TypeaheadComboboxSectionHeader,
  TypeaheadComboboxSeparator,
} from '..';
import { useComboboxState, useMultipleSelectionState } from '../hooks';

describe('Typeahead Combobox', () => {
  test('TypeaheadComboboxMenu throws error if not inside TypeaheadComboboxRoot', () => {
    const TypeaheadComboboxMenuExample: React.FC = () => {
      const [items, setItems] = React.useState<Book[]>(books);

      const comboboxState = useComboboxState<Book>({
        componentId: 'YOUR_TRACKING_ID',
        allItems: books,
        items,
        setItems,
        itemToString: (item: Book) => item.title,
        matcher,
      });

      return (
        <TypeaheadComboboxMenu comboboxState={comboboxState}>
          {items.map((item, index) => (
            <TypeaheadComboboxMenuItem
              key={`book-${item.title}`}
              item={item}
              index={index}
              comboboxState={comboboxState}
            >
              {item.title}
            </TypeaheadComboboxMenuItem>
          ))}
        </TypeaheadComboboxMenu>
      );
    };

    expect(() => render(<TypeaheadComboboxMenuExample />)).toThrow(
      '`TypeaheadComboboxMenu` must be used within `TypeaheadCombobox`',
    );
  });

  test('TypeaheadComboboxInput throws error if not inside TypeaheadComboboxRoot', () => {
    const TypeaheadComboboxInputExample: React.FC = () => {
      const [items, setItems] = React.useState<Book[]>(books);

      const comboboxState = useComboboxState<Book>({
        componentId: 'YOUR_TRACKING_ID',
        allItems: books,
        items,
        setItems,
        itemToString: (item: Book) => item.title,
        matcher,
      });

      return <TypeaheadComboboxInput placeholder="Choose an option" id="book" comboboxState={comboboxState} />;
    };

    expect(() => render(<TypeaheadComboboxInputExample />)).toThrow(
      '`TypeaheadComboboxInput` must be used within `TypeaheadCombobox`',
    );
  });

  test('TypeaheadComboboxMultiSelectInput throws error if not inside TypeaheadComboboxRoot', () => {
    const TypeaheadComboboxMultiSelectInputExample: React.FC = () => {
      const [inputValue, setInputValue] = React.useState('');
      const [selectedItems, setSelectedItems] = React.useState<Book[]>([books[0], books[1]]);
      const items = React.useMemo(() => getFilteredBooks(inputValue), [inputValue]);

      const comboboxState = useComboboxState<Book>({
        componentId: 'YOUR_TRACKING_ID',
        allItems: books,
        items,
        setInputValue,
        matcher,
        itemToString: (item: Book) => item.title,
        multiSelect: true,
        selectedItems,
        setSelectedItems,
      });
      const multipleSelectionState = useMultipleSelectionState<Book>(selectedItems, setSelectedItems, comboboxState);

      return (
        <TypeaheadComboboxMultiSelectInput
          comboboxState={comboboxState}
          multipleSelectionState={multipleSelectionState}
          selectedItems={selectedItems}
          setSelectedItems={setSelectedItems}
          getSelectedItemLabel={(item: Book) => item.title}
        />
      );
    };

    expect(() => render(<TypeaheadComboboxMultiSelectInputExample />)).toThrow(
      '`TypeaheadComboboxMultiSelectInput` must be used within `TypeaheadCombobox`',
    );
  });

  test('TypeaheadComboboxSectionHeader throws error if not inside TypeaheadComboboxRoot', () => {
    expect(() => render(<TypeaheadComboboxSectionHeader />)).toThrow(
      '`TypeaheadComboboxSectionHeader` must be used within `TypeaheadComboboxMenu`',
    );
  });

  test('TypeaheadComboboxSeparator throws error if not inside TypeaheadComboboxRoot', () => {
    expect(() => render(<TypeaheadComboboxSeparator />)).toThrow(
      '`TypeaheadComboboxSeparator` must be used within `TypeaheadComboboxMenu`',
    );
  });

  test('should open menu when input is clicked on', async () => {
    renderSingleSelectTypeahead();
    const input = screen.getByRole('textbox');
    await userEvent.click(input);
    expect(screen.getByRole('listbox')).toBeVisible();
  });

  test('should open menu when associated label is clicked on', async () => {
    renderSingleSelectTypeaheadWithLabel();
    const label = screen.getByText('Favorite book');
    await userEvent.click(label);
    expect(screen.getByRole('listbox')).toBeVisible();
  });

  test('should open and close single-select menu with button', async () => {
    renderSingleSelectTypeahead();
    await openMenuWithButton();
    expect(screen.getByRole('listbox')).toBeVisible();
    await closeMenu();
  });

  test('should open and close multi-select menu with button', async () => {
    renderMultiSelectTypeahead();
    await openMenuWithButton();
    expect(screen.getByRole('listbox')).toBeVisible();
    await closeMenu();
  });

  test('should render tag of selected item after selecting item in multi-select', async () => {
    const text = 'War and Peace';
    renderMultiSelectTypeahead();
    await openMenuWithButton();
    await selectItemByText(text);
    // Search for only the label inside the combobox since it also exists in the menu
    const combobox = screen.getByRole('combobox');
    expect(within(combobox).getAllByText(text)[0]).toBeVisible();
    expect(screen.getByRole('button', { name: 'Remove selected item' })).toBeVisible();
  });

  test('Clear selection button works on single-select', async () => {
    const text = 'War and Peace';
    renderSingleSelectTypeahead();
    await openMenuWithButton();
    await selectItemByText(text);

    const clearButton = screen.getByRole('button', { name: 'Clear selection' });
    await userEvent.click(clearButton);
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(''));
  });

  test('Clear selection button works on multi-select', async () => {
    const text = 'War and Peace';
    renderMultiSelectTypeahead();
    await openMenuWithButton();
    await selectItemByText(text);

    const clearButton = screen.getByRole('button', { name: 'Clear selection' });
    await userEvent.click(clearButton);

    const combobox = screen.getByRole('combobox');
    await waitFor(() => expect(within(combobox).queryByText(text)).toBeNull());
  });

  test('the remove single item button should deselect the item when clicked', async () => {
    const text = 'War and Peace';
    renderMultiSelectTypeahead();
    await openMenuWithButton();
    await selectItemByText(text);

    const removeItemButton = screen.getByRole('button', { name: 'Remove selected item' });
    await userEvent.click(removeItemButton);

    const combobox = screen.getByRole('combobox');
    await waitFor(() => expect(within(combobox).queryByText(text)).toBeNull());
  });

  test('should filter options when typing', async () => {
    renderSingleSelectTypeahead();
    const input = screen.getByRole('textbox');
    await userEvent.click(input);
    await userEvent.type(input, 'mockingbird');
    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(1);
    expect(options[0]).toHaveTextContent('To Kill a Mockingbird');
  });

  test('should be able to navigate options with keyboard', async () => {
    renderSingleSelectTypeahead();
    const input = screen.getByRole('textbox');
    await userEvent.click(input);
    const options = screen.getAllByRole('option');

    // Navigate to first option
    await userEvent.keyboard('{arrowdown}');
    await waitFor(() => expect(options[0]).toHaveAttribute('aria-selected', 'true'));

    // Navigate to second option
    await userEvent.keyboard('{arrowdown}');
    await waitFor(() => {
      expect(options[0]).toHaveAttribute('aria-selected', 'false');
      expect(options[1]).toHaveAttribute('aria-selected', 'true');
    });

    // Navigate back to first option
    await userEvent.keyboard('{arrowup}');
    await waitFor(() => {
      expect(options[0]).toHaveAttribute('aria-selected', 'true');
      expect(options[1]).toHaveAttribute('aria-selected', 'false');
    });
  });

  test('should be able to select an option with keyboard', async () => {
    renderSingleSelectTypeahead();
    const input = screen.getByRole('textbox');
    await userEvent.click(input);
    await userEvent.keyboard('{arrowdown}');
    await userEvent.keyboard('{enter}');
    await waitFor(() => expect(screen.getByRole('textbox')).toHaveValue(books[0].title));
  });

  test('should close menu of multi-select when Escape key is pressed', async () => {
    renderMultiSelectTypeahead();
    await openMenuWithButton();
    await selectItemByText('The Idiot');
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(screen.queryByRole('listbox')).toBeNull());
  });

  test('should remove a selected item on multi-select when backspace key is pressed', async () => {
    const text = 'The Idiot';
    renderMultiSelectTypeahead();
    await openMenuWithButton();
    await selectItemByText(text);

    const combobox = screen.getByRole('combobox');
    expect(within(combobox).getAllByText(text)[0]).toBeVisible();

    await userEvent.keyboard('{Escape}');
    await userEvent.keyboard('{backspace}');
    expect(within(combobox).queryByText(text)).toBeNull();
  });

  test('should render +n badge with remaining items on multi-select after showTagAfterValueCount items have been selected', async () => {
    const NUM_ITEMS = 3;
    const showTagAfterValueCount = 2;
    renderMultiSelectTypeahead({
      preSelectedItems: books.slice(0, NUM_ITEMS),
      multiSelectInputProps: { showTagAfterValueCount },
    });

    const badgeValue = NUM_ITEMS - showTagAfterValueCount;
    const badge = screen.getByText(`+${badgeValue.toString()}`);
    expect(badge).toBeVisible();
  });

  test('should render tooltip on hover when +n badge is shown', async () => {
    const NUM_ITEMS = 22;
    const showTagAfterValueCount = 20; // 20 is default value
    renderMultiSelectTypeahead({ preSelectedItems: books.slice(0, NUM_ITEMS) });

    const badgeValue = NUM_ITEMS - showTagAfterValueCount;
    const badge = screen.getByText(`+${badgeValue.toString()}`);
    await userEvent.hover(badge);

    const tooltipText = books
      .slice(0, NUM_ITEMS)
      .map((book) => book.title)
      .join(', ');
    await waitFor(() => {
      const tooltip = screen.getByRole('tooltip');
      expect(tooltip).toBeInTheDocument();
      expect(tooltip).toHaveTextContent(tooltipText);
    });
  });

  test('should be able to set initial value in single select typeahead', async () => {
    renderSingleSelectTypeahead({ initialInputValue: books[2].title, initialSelectedItem: books[2] });

    expect(screen.getByRole('textbox')).toHaveValue(books[2].title);

    const toggleButton = screen.getByRole('button', { name: 'toggle menu' });
    await userEvent.click(toggleButton);
    expect(screen.getByRole('listbox')).toBeVisible();

    const options = screen.getAllByRole('option');

    await waitFor(() => expect(options[2]).toHaveAttribute('aria-selected', 'true'));
  });

  test('should not show combobox toggle button when showComboboxToggleButton prop is set to false', () => {
    renderSingleSelectTypeahead({ selectInputProps: { showComboboxToggleButton: false } });
    expect(screen.queryByRole('button', { name: 'toggle menu' })).toBeNull();
  });
});
