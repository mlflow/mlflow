import { createIntl } from 'react-intl';
import { I18nUtils } from './I18nUtils';

// see mock for ./loadMessages in setupTests.js

describe('I18nUtils', () => {
  let oldLocation;
  beforeEach(() => {
    oldLocation = window.location;
  });

  afterEach(() => {
    localStorage.clear();
    window.location = oldLocation;
  });

  const setQueryLocale = (locale) => {
    oldLocation = window.location;
    delete window.location;
    window.location = { search: `?l=${locale}` };
  };

  const setLocalStorageLocale = (locale) => {
    localStorage.setItem('locale', locale);
  };

  describe('getCurrentLocale', () => {
    it('should return DEFAULT_LOCALE', () => {
      expect(I18nUtils.getCurrentLocale()).toBe('en');
    });

    it('should prefer locale in l query param over local storage', () => {
      setQueryLocale('fr-CA');
      setLocalStorageLocale('en-US');
      expect(I18nUtils.getCurrentLocale()).toBe('fr-CA');
    });

    it('should not fail for invalid languages', () => {
      const badLocale = 'en_US';
      // example of how it breaks normally
      expect(() => {
        createIntl({ locale: badLocale, defaultLocale: 'en' });
      }).toThrow();

      // we prevent badLocale from getting to creatIntl
      localStorage.setItem('locale', badLocale);
      const locale = I18nUtils.getCurrentLocale();
      expect(locale).toBe('en');
      expect(() => createIntl({ locale, defaultLocale: 'en' })).not.toThrow();
    });

    it('should set locale from query into localStorage', () => {
      setQueryLocale('test-locale');
      expect(I18nUtils.getCurrentLocale()).toBe('test-locale');
    });
    it('should prefer locale from localStorage', () => {
      setLocalStorageLocale('test-locale');
      const locale = I18nUtils.getCurrentLocale();
      expect(locale).toBe('test-locale');
    });
  });

  describe('loadMessages', () => {
    it('should merge values from locale, fallback locale and default locale', async () => {
      expect(await I18nUtils.loadMessages('fr-CA')).toEqual({
        'fr-CA': 'value',
        'fr-FR': 'value',
        en: 'value',
        'top-locale': 'fr-CA',
      });
      expect(await I18nUtils.loadMessages('pt-BR')).toEqual({
        'pt-BR': 'value',
        'pt-PT': 'value',
        en: 'value',
        'top-locale': 'pt-BR',
      });
      expect(await I18nUtils.loadMessages('en-GB')).toEqual({
        'en-GB': 'value',
        en: 'value',
        'top-locale': 'en-GB',
      });
    });

    it('should fallback to base language then default locale for unknown locales', async () => {
      const frResult = await I18nUtils.loadMessages('fr-unknown'); // special mocked locale
      expect(frResult).toEqual({
        'fr-FR': 'value',
        en: 'value',
        'top-locale': 'fr-FR',
      });

      // no base language falls back to default only
      const zzzResult = await I18nUtils.loadMessages('zzz-unknown'); // special mocked locale
      expect(zzzResult).toEqual({
        en: 'value',
        'top-locale': 'en',
      });
    });
  });

  describe('initI18n', () => {
    it('should make messages available to getIntlProviderParams', async () => {
      setLocalStorageLocale('fr-CA');
      await I18nUtils.initI18n();
      expect(I18nUtils.getIntlProviderParams().messages).toEqual({
        'fr-CA': 'value',
        'fr-FR': 'value',
        en: 'value',
        'top-locale': 'fr-CA',
      });
    });
  });
});
