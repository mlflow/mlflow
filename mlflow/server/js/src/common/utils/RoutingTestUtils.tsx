import React from 'react';
import { MemoryRouter, Route, Routes } from './RoutingUtils';

const renderRoute = ({ element, path = '*', pageId, children }: any) => {
  if (children && children.length > 0) {
    return (
      <Route element={element} key={pageId || path} path={path}>
        {children.map(renderRoute)}
      </Route>
    );
  }
  return <Route element={element} key={pageId || path} path={path} />;
};

/**
 * A dummy router to be used in jest tests. Usage:
 *
 * @example
 * ```ts
 *  import { testRoute, TestRouter } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
 *  import { setupTestRouter, waitForRoutesToBeRendered } from '@mlflow/mlflow/src/common/utils/RoutingUtils';
 *
 *  describe('ComponentUnderTest', () => {
 *    it('renders', async () => {
 *      render(<Router history={history} routes={[testRoute(<ComponentUnderTest props={...}/>)]}/>);
 *
 *      expect(...);
 *    });
 *  });
 * ```
 *
 */
export const TestRouter = ({ routes, history, initialEntries }: TestRouterProps) => {
  return (
    <MemoryRouter initialEntries={initialEntries ?? ['/']}>
      <Routes>{routes.map(renderRoute)}</Routes>
    </MemoryRouter>
  );
};

type TestRouteReturnValue = {
  element: React.ReactElement;
  path?: string;
  pageId?: string;
  children?: TestRouteReturnValue[];
};
interface TestRouterProps {
  routes: TestRouteReturnValue[];
  history?: any;
  initialEntries?: string[];
}

export const testRoute = (element: React.ReactNode, path = '*', pageId = '', children = []): TestRouteReturnValue => {
  return { element, path, pageId, children } as any;
};

export const setupTestRouter = () => ({ history: {} });
export const waitForRoutesToBeRendered = async () => {
  return Promise.resolve();
};
