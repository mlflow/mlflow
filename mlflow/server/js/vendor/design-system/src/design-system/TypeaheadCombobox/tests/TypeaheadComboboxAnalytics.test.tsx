import {
  TypeaheadComboboxRoot,
  TypeaheadComboboxMultiSelectInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxCheckboxItem,
  useComboboxState,
  useMultipleSelectionState,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenuItem,
  DesignSystemEventProvider,
  DesignSystemEventProviderAnalyticsEventTypes,
  TypeaheadComboboxFooter,
  TypeaheadComboboxAddButton,
} from '@databricks/design-system';

import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import type { Book } from './testFixtures';
import { books, clickClearButton, selectItemByText } from './testFixtures';

const matcher = (book: Book, query: string) =>
  book.title.toLowerCase().includes(query) || book.author.toLowerCase().includes(query);

function getFilteredBooks(inputValue: string) {
  const lowerCasedInputValue = inputValue.toLowerCase();
  return books.filter(
    (book) =>
      book.title.toLowerCase().includes(lowerCasedInputValue) ||
      book.author.toLowerCase().includes(lowerCasedInputValue),
  );
}

function MultiSelectComboBox() {
  const [inputValue, setInputValue] = React.useState('');
  const [selectedItems, setSelectedItems] = React.useState<Book[]>(books.slice(0, 2));
  const items = React.useMemo(() => getFilteredBooks(inputValue), [inputValue]);
  const itemToString = (item: Book) => item.title;

  const multipleSelectionState = useMultipleSelectionState<Book>(selectedItems, setSelectedItems, {
    componentId: 'book-test',
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true,
    itemToString,
  });

  const comboboxState = useComboboxState<Book>({
    allItems: books,
    items,
    setInputValue,
    matcher,
    itemToString,
    multiSelect: true,
    selectedItems,
    setSelectedItems,
    componentId: 'book-test',
    analyticsEvents: [DesignSystemEventProviderAnalyticsEventTypes.OnValueChange],
    valueHasNoPii: true,
  });

  return (
    <TypeaheadComboboxRoot id="books" multiSelect={true} comboboxState={comboboxState}>
      <TypeaheadComboboxMultiSelectInput
        placeholder="Choose books"
        comboboxState={comboboxState}
        multipleSelectionState={multipleSelectionState}
        selectedItems={selectedItems}
        setSelectedItems={setSelectedItems}
        getSelectedItemLabel={(item: Book) => item.title}
      />
      <TypeaheadComboboxMenu comboboxState={comboboxState}>
        {items.map((item, index) => (
          <TypeaheadComboboxCheckboxItem
            key={`book-${item.title}`}
            item={item}
            index={index}
            comboboxState={comboboxState}
            selectedItems={selectedItems}
          >
            {item.title}
          </TypeaheadComboboxCheckboxItem>
        ))}
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
}

function SingleSelectCombobox({ valueHasNoPii }: { valueHasNoPii?: boolean }) {
  const [items, setItems] = React.useState<Book[]>(books);

  const handleAdd = () => {
    setItems([...items, { author: `New author ${items.length + 1}`, title: `New book ${items.length + 1}` }]);
  };

  const comboboxState = useComboboxState<Book>({
    allItems: books,
    items,
    setItems,
    itemToString: (item: Book) => item.title,
    matcher,
    componentId: 'book-test',
    valueHasNoPii,
  });

  return (
    <TypeaheadComboboxRoot id="book" comboboxState={comboboxState}>
      <TypeaheadComboboxInput comboboxState={comboboxState} />
      <TypeaheadComboboxMenu comboboxState={comboboxState}>
        {items.map((item, index) => (
          <TypeaheadComboboxMenuItem key={`book-${item.title}`} item={item} index={index} comboboxState={comboboxState}>
            {item.title}
          </TypeaheadComboboxMenuItem>
        ))}
        <TypeaheadComboboxFooter>
          <TypeaheadComboboxAddButton componentId="add_book" onClick={handleAdd}>
            Add new book
          </TypeaheadComboboxAddButton>
        </TypeaheadComboboxFooter>
      </TypeaheadComboboxMenu>
    </TypeaheadComboboxRoot>
  );
}

describe('TypeaheadComboboxAnalytics', () => {
  const eventCallback = jest.fn();

  it('emits TypeAheadCombobox Analytics single select events', async () => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <SingleSelectCombobox valueHasNoPii={true} />
      </DesignSystemEventProvider>,
    );

    expect(eventCallback).not.toHaveBeenCalled();

    // Type in the first option and select it.
    const singleSelectCombobox = screen.getByRole('textbox');
    await userEvent.click(singleSelectCombobox);
    await userEvent.type(singleSelectCombobox, 'mockingbird');
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test.input',
      componentType: 'input',
      shouldStartInteraction: false,
      value: undefined,
    });
    await selectItemByText('To Kill a Mockingbird');
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: 'To Kill a Mockingbird',
    });

    // Clear the selection.
    await clickClearButton();
    const combobox = screen.getByRole('combobox');
    await waitFor(() => expect(within(combobox).queryByText('To Kill a Mockingbird')).toBeNull());
    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '',
    });

    // Add a new item and select it.
    await userEvent.click(singleSelectCombobox);
    const addButton = screen.getByRole('button', { name: 'Add new book' });
    await userEvent.click(addButton);
    expect(eventCallback).toHaveBeenCalledTimes(4);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onClick',
      componentId: 'book-test.add_option',
      componentType: 'button',
      shouldStartInteraction: true,
      isInteractionSubject: true,
      event: expect.anything(),
      value: undefined,
    });
    await selectItemByText(`New book ${books.length + 1}`);
    expect(eventCallback).toHaveBeenCalledTimes(5);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: `New book ${books.length + 1}`,
    });
  });

  it('emits TypeAheadCombobox Analytics multi select events', async () => {
    render(
      <DesignSystemEventProvider callback={eventCallback}>
        <MultiSelectComboBox />
      </DesignSystemEventProvider>,
    );

    expect(eventCallback).not.toHaveBeenCalled();

    // Initially, both options are selected. Clear the selections.
    await clickClearButton();
    expect(eventCallback).toHaveBeenCalledTimes(1);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '[]',
    });

    // Select both options.
    await selectItemByText('To Kill a Mockingbird');
    expect(eventCallback).toHaveBeenCalledTimes(2);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '["To Kill a Mockingbird"]',
    });
    await selectItemByText('War and Peace');
    expect(eventCallback).toHaveBeenCalledTimes(3);
    expect(eventCallback).toHaveBeenCalledWith({
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '["To Kill a Mockingbird","War and Peace"]',
    });

    // Deselect option with keyboard.
    await userEvent.click(screen.getByRole('textbox'));
    await userEvent.keyboard('{backspace}');
    expect(eventCallback).toHaveBeenCalledTimes(4);
    expect(eventCallback).toHaveBeenNthCalledWith(4, {
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '["To Kill a Mockingbird"]',
    });

    // Deselect option with mouse.
    await selectItemByText('To Kill a Mockingbird');
    expect(eventCallback).toHaveBeenCalledTimes(5);
    expect(eventCallback).toHaveBeenNthCalledWith(5, {
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '[]',
    });

    // Select an option by typing. multiSelect does not use an input component, so the input event is not fired.
    const multiSelectCombobox = screen.getByRole('textbox');
    await userEvent.click(multiSelectCombobox);
    await userEvent.type(multiSelectCombobox, 'mockingbird');
    await selectItemByText('To Kill a Mockingbird');
    expect(eventCallback).toHaveBeenCalledTimes(6);
    expect(eventCallback).toHaveBeenNthCalledWith(6, {
      eventType: 'onValueChange',
      componentId: 'book-test',
      componentType: 'typeahead_combobox',
      shouldStartInteraction: false,
      value: '["To Kill a Mockingbird"]',
    });
  });
});
