import { fireEvent, within, render, type RenderResult, screen, act, waitFor } from '@testing-library/react';
import React from 'react';
import { IntlProvider } from 'react-intl';
import userEvent from '@testing-library/user-event';
import { DEFAULT_LOCALE } from '../../i18n/loadMessages';
import { DesignSystemProvider } from '@databricks/design-system';

const defaultIntlProviderProps = {
  locale: DEFAULT_LOCALE,
  defaultLocale: DEFAULT_LOCALE,
  messages: {},
};

export function renderWithDesignSystem(ui: React.ReactElement, renderOptions = {}, providerProps = {}): RenderResult {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <IntlProvider {...defaultIntlProviderProps}>
      <DesignSystemProvider {...providerProps}>{children}</DesignSystemProvider>
    </IntlProvider>
  );

  return render(ui, { wrapper, ...renderOptions });
}

export function renderWithIntl(ui: React.ReactElement, renderOptions = {}, providerProps = {}): RenderResult {
  const mergedProviderProps = {
    ...defaultIntlProviderProps,
    ...providerProps,
  };
  const wrapper = ({ children }: { children: React.ReactNode }) => (
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
  await (user ?? userEvent).click(element);
  return (user ?? userEvent).paste(text, { clipboardData: { getData: jest.fn() } } as any);
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

export const findAntdOptionContaining = async (optionText: string) => {
  return await findAntdSelectElement(optionText, '-select-item-option-content');
};

// Function to find the correct antd component based on class name
const findAntdSelectElement = async (optionText: string, endsWith: string) => {
  return await screen.findByText((content, element) => {
    return (
      Boolean(element) &&
      Boolean(Array.from(element?.classList || []).some((x) => x.endsWith(endsWith))) &&
      Boolean(element?.textContent?.includes(optionText))
    );
  });
};

export * from '@testing-library/react';
