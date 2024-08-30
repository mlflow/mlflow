/*
  This file contains utility functions based on react-testing-library@<14,
  dedicated to be used in tests for components that are not yet migrated to react@18.

  Note: those might work with react@18, but it's deprecated and might result in issues/warnings.
*/

import { fireEvent, within, render, type RenderResult, screen, act } from '@testing-library/react';
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
 *
 * @param user Pass this in when the test is using fake timers and the userEvent
 * instance needs to be setup with `advanceTimers` to work properly.
 */
export async function fastFillInput(
  element: HTMLInputElement,
  text: string,
  user?: ReturnType<typeof userEvent.setup>,
) {
  act(() => element.focus());
  return await (user ?? userEvent).paste(text);
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

export const findAntdOption = (optionText: string) => {
  return screen.getByText((content, element) => {
    return (
      Boolean(element) &&
      Boolean(Array.from(element?.classList || []).some((x) => x.endsWith('-select-item-option-content'))) &&
      content === optionText
    );
  });
};

export * from '@testing-library/react';
