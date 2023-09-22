/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';

import {
  HashRouterV5,
  Route,
  Routes,
  CompatRouter,
  LinkV5,
  NavLinkV5,
} from '../../common/utils/RoutingUtils';

import AppErrorBoundary from '../../common/components/error-boundaries/AppErrorBoundary';
import { HomePageDocsUrl, Version } from '../../common/constants';
// @ts-expect-error TS(2307): Cannot find module '../../common/static/home-logo.... Remove this comment to see the full error message
import logo from '../../common/static/home-logo.png';
import ErrorModal from '../../experiment-tracking/components/modals/ErrorModal';
import { CompareModelVersionsPage } from '../../model-registry/components/CompareModelVersionsPage';
import { ModelListPage } from '../../model-registry/components/ModelListPage';
import { ModelPage } from '../../model-registry/components/ModelPage';
import { ModelVersionPage } from '../../model-registry/components/ModelVersionPage';
import {
  compareModelVersionsPageRoute,
  modelListPageRoute,
  modelPageRoute,
  modelSubpageRoute,
  modelSubpageRouteWithName,
  modelVersionPageRoute,
} from '../../model-registry/routes';
import RoutePaths from '../routes';
import './App.css';
import CompareRunPage from './CompareRunPage';
import { HomePage } from './HomePage';
import { MetricPage } from './MetricPage';
import { PageNotFoundView } from '../../common/components/PageNotFoundView';
import { RunPage } from './RunPage';
import { DirectRunPage } from './DirectRunPage';

const isExperimentsActive = (match: any, location: any) => {
  // eslint-disable-next-line prefer-const
  return match && !location.pathname.includes('models');
};

let mlflowHashRouting = false;

export function setMLFlowHashRouting() {
  mlflowHashRouting = true;
}

const classNames = {
  activeNavLink: { borderBottom: '4px solid #43C9ED' },
};

// eslint-disable-next-line no-unused-vars
const InteractionTracker = ({ children }: any) => children;

class App extends Component {
  render() {
    const marginRight = 24;
    return (
      <HashRouterV5
        basename={mlflowHashRouting ? '/mlflow' : undefined}
        hashType={mlflowHashRouting ? 'noslash' : undefined}
      >
        {/* This layer enables intercompatibility between react-router APIs v5 and v6 */}
        {/* TODO: Remove after migrating to react-router v6 */}
        <CompatRouter>
          <div style={{ height: '100%' }}>
            <ErrorModal />
            {/* @ts-expect-error TS(4111): Property 'HIDE_HEADER' comes from an index signatu... Remove this comment to see the full error message */}
            {process.env.HIDE_HEADER === 'true' ? null : (
              <header className='App-header'>
                <div className='mlflow-logo'>
                  <LinkV5 to={RoutePaths.rootRoute} className='App-mlflow'>
                    <img className='mlflow-logo' alt='MLflow' src={logo} />
                  </LinkV5>
                  <span className={'mlflow-version'}>{Version}</span>
                </div>
                <div className='header-route-links'>
                  <NavLinkV5
                    strict
                    to={RoutePaths.rootRoute}
                    css={{ marginRight }}
                    activeStyle={classNames.activeNavLink}
                    isActive={isExperimentsActive}
                    className='header-nav-link'
                  >
                    <div className='experiments'>
                      <span>Experiments</span>
                    </div>
                  </NavLinkV5>
                  <NavLinkV5
                    strict
                    to={modelListPageRoute}
                    css={{ marginRight }}
                    activeStyle={classNames.activeNavLink}
                    className='header-nav-link header-nav-link-models'
                  >
                    <div className='models'>
                      <span>Models</span>
                    </div>
                  </NavLinkV5>
                </div>
                <div className='header-links'>
                  <a href={'https://github.com/mlflow/mlflow'} css={{ marginRight }}>
                    <div className='github'>
                      <span>GitHub</span>
                    </div>
                  </a>
                  <a href={HomePageDocsUrl} css={{ marginRight }}>
                    <div className='docs'>
                      <span>Docs</span>
                    </div>
                  </a>
                </div>
              </header>
            )}
            <AppErrorBoundary service='mlflow'>
              <InteractionTracker>
                {/* The block below contains React Router v6 routes */}
                <Routes>
                  <Route
                    path={RoutePaths.compareExperimentsSearchPageRoute}
                    element={<HomePage />}
                  />
                  <Route path={RoutePaths.experimentPageSearchRoute} element={<HomePage />} />
                  <Route path={RoutePaths.experimentPageRoute} element={<HomePage />} />
                  <Route path={RoutePaths.rootRoute} element={<HomePage />} />
                  <Route path={RoutePaths.runPageWithArtifactSelectedRoute} element={<RunPage />} />
                  <Route path={RoutePaths.runPageRoute} element={<RunPage />} />
                  <Route path={RoutePaths.directRunPageRoute} element={<DirectRunPage />} />

                  <Route path={RoutePaths.metricPageRoute} element={<MetricPage />} />
                  <Route path={RoutePaths.compareRunPageRoute} element={<CompareRunPage />} />

                  <Route path={modelListPageRoute} element={<ModelListPage />} />
                  <Route path={modelVersionPageRoute} element={<ModelVersionPage />} />
                  <Route path={modelPageRoute} element={<ModelPage />} />
                  <Route path={modelSubpageRoute} element={<ModelPage />} />
                  <Route path={modelSubpageRouteWithName} element={<ModelPage />} />
                  <Route
                    path={compareModelVersionsPageRoute}
                    element={<CompareModelVersionsPage />}
                  />
                  <Route path='/*' element={<PageNotFoundView />} />
                </Routes>
                {/* End of React Router v6 routes */}
              </InteractionTracker>
            </AppErrorBoundary>
          </div>
        </CompatRouter>
      </HashRouterV5>
    );
  }
}

const mapStateToProps = (state: any) => {
  return {
    experiments: Object.values(state.entities.experimentsById),
  };
};

export default connect(mapStateToProps)(App);
