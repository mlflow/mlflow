import '@databricks/design-system/dist/index.css';
import { designSystemDecorator } from './decorators/design-system';
import '../src/index.css';
import { withIntlDecorator } from './decorators/with-intl';
import { withRouterDecorator } from './decorators/with-router';
import { withReduxDecorator } from './decorators/with-redux';

export const parameters = {
  controls: {
    matchers: {
      color: /(background|color)$/i,
      date: /Date$/,
    },
  },
};

export const decorators = [
  designSystemDecorator,
  withIntlDecorator,
  withRouterDecorator,
  withReduxDecorator,
];
