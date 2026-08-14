import { IntlProvider } from 'react-intl';

const defaultIntlProps = {
  messages: {},
  locale: 'en',
  onError: (e) => {
    // Omit missing translation errors in storybook
    if (e.code === 'MISSING_TRANSLATION') {
      return null;
    }
    throw e;
  },
};

/**
 * Enabling react-intl capabilities to stories by wrapping the story
 * with the custom Intl Provider.
 *
 * Basic usage:
 *
 * export default {
 *   title: 'Story/Path',
 *   component: Component,
 *   parameters: {
 *     withIntl: true
 *   }
 * };
 *
 * Usage with changed IntlProvider settings:
 *
 * export default {
 *   title: 'Story/Path',
 *   component: Component,
 *   parameters: {
 *     withIntl: {
 *       locale: 'jp',
 *       messages: {
 *         foo: 'bar'
 *       },
 *     },
 *   },
 * };
 */
export const withIntlDecorator = (Story, { parameters }) => {
  if (parameters.withIntl) {
    const intlProps = {
      ...defaultIntlProps,
      ...(typeof parameters.withIntl === 'object' ? parameters.withIntl : {}),
    };

    return (
      <IntlProvider {...intlProps}>
        <Story />
      </IntlProvider>
    );
  }

  return <Story />;
};
