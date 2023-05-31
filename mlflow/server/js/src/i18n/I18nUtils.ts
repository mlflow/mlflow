/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { createIntlCache, createIntl } from 'react-intl';
import { DEFAULT_LOCALE, loadMessages } from './loadMessages';
import { useEffect, useState } from 'react';

const FALLBACK_LOCALES: Record<string, string> = {
  es: 'es-ES',
  fr: 'fr-FR',
  pt: 'pt-PT',
  ja: 'ja-JP',
  kr: 'kr-KR',
  it: 'it-IT',
  de: 'de-DE',
  zh: 'zh-CN',
};

const loadedMessages: Record<string, any> = {};

const cache = createIntlCache();

export const I18nUtils = {
  async initI18n() {
    const locale = I18nUtils.getCurrentLocale();
    await I18nUtils.loadMessages(locale);
    return { locale, messages: loadedMessages[locale] };
  },

  getIntlProviderParams() {
    const locale = I18nUtils.getCurrentLocale();
    return {
      locale,
      messages: loadedMessages[locale] || {},
    };
  },

  /**
   * When intl object is used entirely outside of React (e.g., Backbone Views) then
   * this method can be used to get the intl object.
   */
  createIntlWithLocale() {
    const params = I18nUtils.getIntlProviderParams();
    return createIntl({ locale: params.locale, messages: params.messages }, cache);
  },

  getCurrentLocale() {
    const queryParams = new URLSearchParams(window.location.search);
    const getLocale = () => {
      const langFromQuery = queryParams.get('l');
      if (langFromQuery) {
        window.localStorage.setItem('locale', langFromQuery);
      }
      return window.localStorage.getItem('locale') || DEFAULT_LOCALE;
    };
    const locale = getLocale();

    // _ in the locale causes createIntl to throw, so convert to default locale
    if (locale.includes('_')) {
      return DEFAULT_LOCALE;
    }
    return locale;
  },

  /* Gets the locale to fall back on if messages are missing */
  getFallbackLocale(locale: string) {
    const lang = locale.split('-')[0];
    const fallback = FALLBACK_LOCALES[lang];
    return fallback === lang ? undefined : fallback;
  },

  async loadMessages(locale: string) {
    const locales = [
      locale === DEFAULT_LOCALE ? undefined : DEFAULT_LOCALE,
      I18nUtils.getFallbackLocale(locale),
      locale,
    ].filter(Boolean);
    const results = await Promise.all(locales.map(loadMessages));
    loadedMessages[locale] = Object.assign({}, ...results);
    return loadedMessages[locale];
  },
};

export type UseI18nInitResult = {
  locale: string;
  messages: Record<string, any>;
};

/**
 * Ensure initialization of i18n subsystem and return
 * an object with current locale and messages storage.
 *
 * The returned value will be null before initialization.
 *
 * This hook is intended to be used once in the top-level components.
 */
export const useI18nInit = () => {
  const [intlState, setIntlState] = useState<UseI18nInitResult | null>(null);
  useEffect(() => {
    I18nUtils.initI18n()
      .then((initializedIntlState) => {
        setIntlState(initializedIntlState);
      })
      .catch((error) => {
        // Fall back to the defaults if loading translation fails
        setIntlState(I18nUtils.getIntlProviderParams());
      });
  }, []);

  return intlState;
};
