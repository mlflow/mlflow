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
  // No compiled messages for the default locale — react-intl renders the
  // inline `defaultMessage` from each <FormattedMessage> / formatMessage call.
  if (locale === DEFAULT_LOCALE) {
    return {};
  }

  try {
    return (await import(`../lang/compiled/${locale}.json`)).default;
  } catch (e) {
    return {};
  }
}
