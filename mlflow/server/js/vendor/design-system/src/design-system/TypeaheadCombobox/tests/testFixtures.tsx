import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import React from 'react';

import {
  TypeaheadComboboxCheckboxItem,
  TypeaheadComboboxInput,
  TypeaheadComboboxMenu,
  TypeaheadComboboxMenuItem,
  TypeaheadComboboxMultiSelectInput,
  TypeaheadComboboxRoot,
} from '..';
import { DesignSystemProvider } from '../../DesignSystemProvider';
import { FormUI } from '../../FormV2';
import type { TypeaheadComboboxInputProps } from '../TypeaheadComboboxInput';
import type { TypeaheadComboboxMultiSelectInputProps } from '../TypeaheadComboboxMultiSelectInput';
import { useComboboxState, useMultipleSelectionState } from '../hooks';

export interface Book {
  author: string;
  title: string;
}

export interface renderMultiSelectTypeaheadProps {
  preSelectedItems?: Book[];
  multiSelectInputProps?: Partial<TypeaheadComboboxMultiSelectInputProps<Book>>;
}

export interface renderSingleSelectTypeaheadProps {
  initialInputValue?: string;
  initialSelectedItem?: Book;
  selectInputProps?: Partial<TypeaheadComboboxInputProps<Book>>;
}

export const books: Book[] = [
  { author: 'Harper Lee', title: 'To Kill a Mockingbird' },
  { author: 'Lev Tolstoy', title: 'War and Peace' },
  { author: 'Fyodor Dostoyevsy', title: 'The Idiot' },
  { author: 'Oscar Wilde', title: 'A Picture of Dorian Gray' },
  { author: 'George Orwell', title: '1984' },
  { author: 'Jane Austen', title: 'Pride and Prejudice' },
  { author: 'Marcus Aurelius', title: 'Meditations' },
  { author: 'Fyodor Dostoevsky', title: 'The Brothers Karamazov' },
  { author: 'Lev Tolstoy', title: 'Anna Karenina' },
  { author: 'Fyodor Dostoevsky', title: 'Crime and Punishment' },
  { author: 'F. Scott Fitzgerald', title: 'The Great Gatsby' },
  { author: 'Emily Bronte', title: 'Wuthering Heights' },
  { author: 'J.R.R. Tolkien', title: 'The Lord of the Rings' },
  { author: 'Bram Stoker', title: 'Dracula' },
  { author: 'Charles Dickens', title: 'Great Expectations' },
  { author: 'Herman Melville', title: 'Moby Dick' },
  { author: 'J.D. Salinger', title: 'The Catcher in the Rye' },
  { author: 'Louisa May Alcott', title: 'Little Women' },
  { author: 'Mark Twain', title: 'Adventures of Huckleberry Finn' },
  { author: 'William Golding', title: 'Lord of the Flies' },
  { author: 'George Orwell', title: 'Animal Farm' },
  { author: 'Charles Dickens', title: 'A Tale of Two Cities' },
  { author: 'George Eliot', title: 'Middlemarch' },
  { author: 'Miguel de Cervantes', title: 'Don Quixote' },
  { author: 'Aldous Huxley', title: 'Brave New World' },
  { author: 'Nathaniel Hawthorne', title: 'The Scarlet Letter' },
  { author: 'Ray Bradbury', title: 'Fahrenheit 451' },
];

export const matcher = (book: Book, query: string) =>
  book.title.toLowerCase().includes(query) || book.author.toLowerCase().includes(query);

export const getFilteredBooks = (inputValue: string) => {
  const lowerCasedInputValue = inputValue.toLowerCase();
  return books.filter(
    (book) =>
      book.title.toLowerCase().includes(lowerCasedInputValue) ||
      book.author.toLowerCase().includes(lowerCasedInputValue),
  );
};

export const renderSingleSelectTypeahead = ({
  initialSelectedItem,
  initialInputValue,
  selectInputProps = {},
}: renderSingleSelectTypeaheadProps = {}) => {
  const SingleSelectTypeahead: React.FC = () => {
    const [items, setItems] = React.useState<Book[]>(books);

    const comboboxState = useComboboxState<Book>({
      componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_tests_testfixtures.tsx_96',
      allItems: books,
      items,
      setItems,
      itemToString: (item: Book) => item.title,
      matcher,
      initialInputValue,
      initialSelectedItem,
    });

    return (
      <DesignSystemProvider>
        <TypeaheadComboboxRoot id="book" comboboxState={comboboxState}>
          <TypeaheadComboboxInput placeholder="Choose an option" comboboxState={comboboxState} {...selectInputProps} />
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
        </TypeaheadComboboxRoot>
      </DesignSystemProvider>
    );
  };

  render(<SingleSelectTypeahead />);
};

export const renderMultiSelectTypeahead = ({
  preSelectedItems = [],
  multiSelectInputProps = {},
}: renderMultiSelectTypeaheadProps = {}) => {
  const MultiSelectTypeahead: React.FC = () => {
    const [inputValue, setInputValue] = React.useState('');
    const [selectedItems, setSelectedItems] = React.useState<Book[]>(preSelectedItems);
    const items = React.useMemo(() => getFilteredBooks(inputValue), [inputValue]);

    const comboboxState = useComboboxState<Book>({
      componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_tests_testfixtures.tsx_146',
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
      <DesignSystemProvider>
        <TypeaheadComboboxRoot id="books" multiSelect={true} comboboxState={comboboxState}>
          <TypeaheadComboboxMultiSelectInput
            placeholder={selectedItems.length ? '' : 'Choose books'}
            comboboxState={comboboxState}
            multipleSelectionState={multipleSelectionState}
            selectedItems={selectedItems}
            setSelectedItems={setSelectedItems}
            getSelectedItemLabel={(item: Book) => item.title}
            {...multiSelectInputProps}
          />
          <TypeaheadComboboxMenu comboboxState={comboboxState} width={300}>
            {items.map((item, index) => (
              <TypeaheadComboboxCheckboxItem
                key={`book-${item.title}`}
                item={item}
                index={index}
                comboboxState={comboboxState}
                selectedItems={selectedItems}
                textOverflowMode="ellipsis"
              >
                {item.title}
              </TypeaheadComboboxCheckboxItem>
            ))}
          </TypeaheadComboboxMenu>
        </TypeaheadComboboxRoot>
      </DesignSystemProvider>
    );
  };

  render(<MultiSelectTypeahead />);
};

export const renderSingleSelectTypeaheadWithLabel = ({
  initialSelectedItem,
  initialInputValue,
  selectInputProps = {},
}: renderSingleSelectTypeaheadProps = {}) => {
  const SingleSelectTypeahead: React.FC = () => {
    const [items, setItems] = React.useState<Book[]>(books);

    const comboboxState = useComboboxState<Book>({
      componentId: 'codegen_design-system_src_design-system_typeaheadcombobox_tests_testfixtures.tsx_96',
      allItems: books,
      items,
      setItems,
      itemToString: (item: Book) => item.title,
      matcher,
      initialInputValue,
      initialSelectedItem,
    });

    return (
      <DesignSystemProvider>
        <FormUI.Label htmlFor="book" {...comboboxState.getLabelProps()}>
          Favorite book
        </FormUI.Label>
        <TypeaheadComboboxRoot id="book" comboboxState={comboboxState}>
          <TypeaheadComboboxInput placeholder="Choose an option" comboboxState={comboboxState} {...selectInputProps} />
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
        </TypeaheadComboboxRoot>
      </DesignSystemProvider>
    );
  };

  render(<SingleSelectTypeahead />);
};

export const openMenuWithButton = async () => {
  const toggleButton = screen.getByRole('button', { name: 'toggle menu' });
  await userEvent.click(toggleButton);
  expect(screen.getByRole('listbox')).toBeVisible();
};

export const closeMenu = async () => {
  const toggleButton = screen.getByRole('button');
  await userEvent.click(toggleButton);
  expect(screen.queryByRole('listbox')).toBeNull();
};

export const selectItemByText = async (text: string) => {
  await userEvent.click(screen.getByRole('option', { name: text }));
};

export const clickClearButton = async () => {
  const clearButton = screen.getByRole('button', { name: 'Clear selection' });
  await userEvent.click(clearButton);
};
