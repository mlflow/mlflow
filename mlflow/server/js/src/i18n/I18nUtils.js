import en from './lang/en.json';

export const DEFAULT_LOCALE = 'en';

const LOCALE_TO_MESSAGES = {
  en: en,
};

export class I18nUtils {
  static getCurrentLocale() {
    const { language } = navigator;
    // _ in the locale causes createIntl to throw, so convert to default locale
    if (language.includes('_')) {
      return DEFAULT_LOCALE;
    }
    return language.split('-')[0];
  }

  static getMessages(locale) {
    return LOCALE_TO_MESSAGES[locale];
  }
}
