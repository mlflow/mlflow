/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';
import { createIntl } from 'react-intl';
import { I18nUtils, useI18nInit } from './I18nUtils';
import { mount } from 'enzyme';
import { act } from 'react-dom/test-utils';

// see mock for ./loadMessages in setupTests.js

describe('I18nUtils', () => {
  let oldLocation: any;
  beforeEach(() => {
    oldLocation = window.location;
  });

  afterEach(() => {
    localStorage.clear();
    window.location = oldLocation;
  });

  const setQueryLocale = (locale: any) => {
    oldLocation = window.location;
    // @ts-expect-error TS(2790): The operand of a 'delete' operator must be optiona... Remove this comment to see the full error message
    delete window.location;
    // @ts-expect-error TS(2322): Type '{ search: string; }' is not assignable to ty... Remove this comment to see the full error message
    window.location = { search: `?l=${locale}` };
  };

  const setLocalStorageLocale = (locale: any) => {
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

  describe('useI18nInit', () => {
    const mockLookupFn = jest.fn();

    const TestComponent = () => {
      const result = useI18nInit();
      mockLookupFn(result);
      return null;
    };

    beforeEach(() => {
      mockLookupFn.mockClear();
    });

    it('waits for loading messages', async () => {
      const wrapper = mount(React.createElement(TestComponent));

      // Initial call - no messages loaded yet
      expect(mockLookupFn).toBeCalledWith(null);

      // Flush all promises and update the component
      await act(async () => {
        await new Promise(setImmediate);
        wrapper.update();
      });

      // Now we should have mock messages (defined in setupTests.js)
      expect(mockLookupFn).toBeCalledWith({
        locale: 'en',
        messages: { en: 'value', 'top-locale': 'en' },
      });
    });

    it('falls back to the default value when necessary', async () => {
      // choose a different locale for this test
      window.localStorage.setItem('locale', 'de-DE');

      const errorThrown = new Error('failing translation load');

      const originalI18nUtils = { ...I18nUtils };
      I18nUtils.initI18n = jest.fn().mockRejectedValue(errorThrown);

      const wrapper = mount(React.createElement(TestComponent));

      // Flush all promises and update the component
      await act(async () => {
        await new Promise(setImmediate);
        wrapper.update();
      });

      // Now we should have fallback messages (empty set)
      expect(mockLookupFn).toBeCalledWith({
        locale: 'de-DE',
        messages: {},
      });

      I18nUtils.initI18n = originalI18nUtils.initI18n;
    });
  });
});
