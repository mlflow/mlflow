/**
 * This file separates the webpack require.context from the rest of the i18n message loading
 * so that it can be mocked in tests.
 */
// Import default locale statically to avoid slowing page load for existing users
import defaultMessages from './default/en.json';
import { generateENXA } from '@formatjs/cli/src/pseudo_locale';

const messagesContext = require.context('./lang', true, /\.json$/, 'lazy');
const messagePaths = messagesContext.keys();

export const DEFAULT_LOCALE = 'en';

export async function loadMessages(locale) {
  if (locale === DEFAULT_LOCALE) {
    return defaultMessages;
  }
  if (locale === 'dev') {
    const pseudoMessages = {};
    Object.entries(defaultMessages).forEach(
      ([key, value]) => (pseudoMessages[key] = generateENXA(value)),
    );
    return pseudoMessages;
  }

  const path = messagePaths.find((x) => x === `./${locale}.json`);
  if (path) {
    return messagesContext(path);
  }
  return {};
}
