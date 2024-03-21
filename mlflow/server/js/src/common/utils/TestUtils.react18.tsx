/*
  This file contains utility functions based on react-testing-library@>=14,
  dedicated to be used in tests for components migrated to react@18.

  Will NOT work with react@17.
*/

import {
  fireEvent,
  within,
  render,
  type RenderResult,
  screen,
  act,
  waitFor,
} from '@testing-library/react-for-react-18';
import React from 'react';
import { IntlProvider } from 'react-intl';
import userEvent from '@testing-library/user-event-14';
import { DEFAULT_LOCALE } from '../../i18n/loadMessages';

const defaultProviderProps = {
  locale: DEFAULT_LOCALE,
  defaultLocale: DEFAULT_LOCALE,
  messages: {},
};

export function renderWithIntl(ui: React.ReactElement, renderOptions = {}, providerProps = {}): RenderResult {
  const mergedProviderProps = {
    ...defaultProviderProps,
    ...providerProps,
  };
  const wrapper: React.JSXElementConstructor<{ children: React.ReactElement }> = ({ children }) => (
    <IntlProvider {...mergedProviderProps}>{children}</IntlProvider>
  );

  return render(ui, { wrapper, ...renderOptions });
}
/**
 * userEvent.type() can be quite slow, let's use userEvent.paste()
 * to improve testing performance
 */
export async function fastFillInput(element: HTMLInputElement, text: string) {
  await userEvent.click(element);
  return userEvent.paste(text, { clipboardData: { getData: jest.fn() } } as any);
}

export const selectAntdOption = async (container: HTMLElement, optionText: string) => {
  await act(async () => {
    fireEvent.mouseDown(within(container).getByRole('combobox'));
  });
  const optionElement = findAntdOption(optionText);
  act(() => {
    fireEvent.click(optionElement);
  });
};

export const selectAntdOptionByText = async (container: HTMLElement, optionText: string) => {
  // open the select component
  await userEvent.click(within(container).getByRole('combobox'));
  // select option
  await userEvent.click(await findAntdOptionContaining(optionText));
  // wait for the option selected to be reflected in the UI
  await waitFor(async () => {
    expect(await findAntdSelectElement(optionText, '-select-selection-item')).toBeInTheDocument();
  });
};

export const findAntdOption = (optionText: string) => {
  return screen.getByText((content, element) => {
    return (
      Boolean(element) &&
      Boolean(Array.from(element?.classList || []).some((x) => x.endsWith('-select-item-option-content'))) &&
      content === optionText
    );
  });
};

export const findAntdOptionContaining = async (optionText: string) => {
  return await findAntdSelectElement(optionText, '-select-item-option-content');
};

// Function to find the correct antd component based on class name
export const findAntdSelectElement = async (optionText: string, endsWith: string) => {
  return await screen.findByText((content, element) => {
    return (
      Boolean(element) &&
      Boolean(Array.from(element?.classList || []).some((x) => x.endsWith(endsWith))) &&
      Boolean(element?.textContent?.includes(optionText))
    );
  });
};

export * from '@testing-library/react-for-react-18';
