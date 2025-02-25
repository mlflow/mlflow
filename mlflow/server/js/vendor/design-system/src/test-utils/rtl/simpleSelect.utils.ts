/* eslint-disable @databricks/no-restricted-globals-with-module */
import { fireEvent, screen } from '@testing-library/react';

// Helpers to get the selected option from the trigger
const getSelectedOptionLabelFromTrigger = (name?: string | RegExp): string | null => {
  return screen.getByRole('combobox', { name }).textContent;
};

const getSelectedOptionValueFromTrigger = (name?: string | RegExp): string | null => {
  return screen.getByRole('combobox', { name }).getAttribute('value');
};

const getSelectedOptionFromTrigger = (name?: string | RegExp): { label: string | null; value: string | null } => {
  const label = getSelectedOptionLabelFromTrigger(name);
  const value = getSelectedOptionValueFromTrigger(name);
  return { label, value };
};

const expectSelectedOptionFromTriggerToBe = (label: string | RegExp, name?: string | RegExp): void => {
  expect(getSelectedOptionLabelFromTrigger(name)).toBe(label);
};

const toggleSelect = (name?: string | RegExp): void => {
  fireEvent.click(screen.getByRole('combobox', { name }));
};

const expectSelectToBeOpen = (): void => {
  expect(screen.queryAllByRole('option')).not.toHaveLength(0);
};

const expectSelectToBeClosed = (): void => {
  expect(screen.queryAllByRole('option')).toHaveLength(0);
};

// Generic helpers for when the select is open
const getOptionsLength = (): number => {
  return screen.getAllByRole('option').length;
};

const getAllOptions = (): (string | null)[] => {
  return screen.getAllByRole('option').flatMap((option) => option.textContent);
};

const expectOptionsLengthToBe = (length: number): void => {
  expect(getOptionsLength()).toBe(length);
};

const getUnselectedOption = (label: string | RegExp): HTMLElement => {
  return screen.getByRole('option', { name: label, selected: false });
};

const getSelectedOption = (label: string | RegExp): HTMLElement => {
  return screen.getByRole('option', { name: label, selected: false });
};

const getOption = (label: string | RegExp): HTMLElement => {
  return screen.getByRole('option', { name: label });
};

const selectOption = (label: string | RegExp): void => {
  fireEvent.click(screen.getByRole('option', { name: label }));
};

const expectSelectedOptionToBe = (label: string | RegExp): void => {
  const options = screen.getAllByRole('option');
  const selectedOption = options.find((option) => option.getAttribute('aria-selected') === 'true');
  expect(selectedOption).toHaveTextContent(label);
};

export const simpleSelectTestUtils = {
  getSelectedOptionLabelFromTrigger,
  getSelectedOptionValueFromTrigger,
  getSelectedOptionFromTrigger,
  expectSelectedOptionFromTriggerToBe,
  toggleSelect,
  expectSelectToBeOpen,
  expectSelectToBeClosed,
  getOptionsLength,
  getAllOptions,
  expectOptionsLengthToBe,
  getUnselectedOption,
  getSelectedOption,
  getOption,
  selectOption,
  expectSelectedOptionToBe,
};
