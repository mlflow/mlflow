import { createIntlCache, createIntl } from 'react-intl';
import { DEFAULT_LOCALE, loadMessages } from './loadMessages';

const FALLBACK_LOCALES = {
  fr: 'fr-FR',
  pt: 'pt-PT',
};

const loadedMessages = {};

const cache = createIntlCache();

export const I18nUtils = {
  async initI18n() {
    const locale = I18nUtils.getCurrentLocale();
    await I18nUtils.loadMessages(locale);
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
  getFallbackLocale(locale) {
    const lang = locale.split('-')[0];
    const fallback = FALLBACK_LOCALES[lang];
    return fallback === lang ? undefined : fallback;
  },

  async loadMessages(locale) {
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
