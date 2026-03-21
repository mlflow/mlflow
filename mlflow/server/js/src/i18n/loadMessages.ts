/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/**
 * This file separates the webpack require.context from the rest of the i18n message loading
 * so that it can be mocked in tests.
 */
export const DEFAULT_LOCALE = 'en';

export async function loadMessages(locale: any) {
  if (locale === DEFAULT_LOCALE) {
    return {};
  }
  if (locale === 'dev') {
    const pseudoMessages = {};
    const defaultMessages = await import('../lang/default/en.json');
    const { generateENXA } = await import('@formatjs/cli/src/pseudo_locale');
    Object.entries(defaultMessages).forEach(
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      ([key, value]) => (pseudoMessages[key] = generateENXA(value)),
    );
    return pseudoMessages;
  }

  try {
    return (await import(`../lang/compiled/${locale}.json`)).default;
  } catch (e) {
    return {};
  }
}
