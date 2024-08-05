import PageNotFoundView from './components/PageNotFoundView';
import LoginView from './components/LoginView';
import LoginLoadingView from './components/LoginLoadingView';
import { createMLflowRoutePath, createRouteElement } from './utils/RoutingUtils';

/**
 * Common route definitions. For the time being it's 404 page only.
 */
export const getRouteDefs = () => [
  {
    path: createMLflowRoutePath('/*'),
    element: createRouteElement(PageNotFoundView),
    pageId: 'mlflow.common.not-found',
  },
  {
    path: createMLflowRoutePath('/sso-login'),
    element: createRouteElement(LoginView),
    pageId: 'mlflow.common.sso-login',
  },
  {
    path: createMLflowRoutePath('/sso-login-loading'),
    element: createRouteElement(LoginLoadingView),
    pageId: 'mlflow.common.sso-login-loading',
  }
];