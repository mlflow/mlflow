/* eslint-disable @databricks/no-restricted-globals-with-module */
import { fireEvent, screen } from '@testing-library/react';
// Helpers to get the selected option from the trigger
const getSelectedOptionLabelFromTrigger = (name) => {
    return screen.getByRole('combobox', { name }).textContent;
};
const getSelectedOptionValueFromTrigger = (name) => {
    return screen.getByRole('combobox', { name }).getAttribute('value');
};
const getSelectedOptionFromTrigger = (name) => {
    const label = getSelectedOptionLabelFromTrigger(name);
    const value = getSelectedOptionValueFromTrigger(name);
    return { label, value };
};
const expectSelectedOptionFromTriggerToBe = (label, name) => {
    expect(getSelectedOptionLabelFromTrigger(name)).toBe(label);
};
const toggleSelect = (name) => {
    fireEvent.click(screen.getByRole('combobox', { name }));
};
const expectSelectToBeOpen = () => {
    expect(screen.queryAllByRole('option')).not.toHaveLength(0);
};
const expectSelectToBeClosed = () => {
    expect(screen.queryAllByRole('option')).toHaveLength(0);
};
// Generic helpers for when the select is open
const getOptionsLength = () => {
    return screen.getAllByRole('option').length;
};
const getAllOptions = () => {
    return screen.getAllByRole('option').flatMap((option) => option.textContent);
};
const expectOptionsLengthToBe = (length) => {
    expect(getOptionsLength()).toBe(length);
};
const getUnselectedOption = (label) => {
    return screen.getByRole('option', { name: label, selected: false });
};
const getSelectedOption = (label) => {
    return screen.getByRole('option', { name: label, selected: false });
};
const getOption = (label) => {
    return screen.getByRole('option', { name: label });
};
const selectOption = (label) => {
    fireEvent.click(screen.getByRole('option', { name: label }));
};
const expectSelectedOptionToBe = (label) => {
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
//# sourceMappingURL=simpleSelect.utils.js.map