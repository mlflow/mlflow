import { StaticRouter } from 'react-router';

/**
 * Adds router capabilities to stories by wrapping the story
 * with Static Router.
 *
 * Basic usage:
 *
 * export default {
 *   title: 'Story/Path',
 *   component: Component,
 *   parameters: {
 *     withRouter: true
 *   }
 * };
 *
 * Usage with changing location:
 *
 * export default {
 *   title: 'Story/Path',
 *   component: Component,
 *   parameters: {
 *     withRouter: {
 *       location: '/some/location',
 *     },
 *   },
 * };
 */
export const withRouterDecorator = (Story, { parameters }) => {
  if (parameters.withRouter) {
    const routerProps = typeof parameters.withRouter === 'object' ? parameters.withRouter : {};
    return (
      <StaticRouter location='/' {...routerProps}>
        <Story />
      </StaticRouter>
    );
  }

  return <Story />;
};
