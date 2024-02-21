/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';

import { HashRouterV5, Route, Routes, CompatRouter, LinkV5, NavLinkV5 } from '../../common/utils/RoutingUtils';

import AppErrorBoundary from '../../common/components/error-boundaries/AppErrorBoundary';
import { HomePageDocsUrl, Version } from '../../common/constants';
import logo from '../../common/static/home-logo.png';
import ErrorModal from '../../experiment-tracking/components/modals/ErrorModal';
import { CompareModelVersionsPage } from '../../model-registry/components/CompareModelVersionsPage';
import { ModelListPage } from '../../model-registry/components/ModelListPage';
import { ModelPage } from '../../model-registry/components/ModelPage';
import { ModelVersionPage } from '../../model-registry/components/ModelVersionPage';
import { ModelRegistryRoutes, ModelRegistryRoutePaths } from '../../model-registry/routes';
import ExperimentTrackingRoutes, { RoutePaths as ExperimentTrackingRoutePaths } from '../routes';
import './App.css';
import CompareRunPage from './CompareRunPage';
import HomePage from './HomePage';
import { MetricPage } from './MetricPage';
import { PageNotFoundView } from '../../common/components/PageNotFoundView';
import { RunPage } from './RunPage';
import { DirectRunPage } from './DirectRunPage';
import { shouldEnableDeepLearningUI } from '../../common/utils/FeatureUtils';
import { DarkThemeSwitch } from '../../common/components/DarkThemeSwitch';

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

interface AppProps {
  isDarkTheme?: boolean;
  setIsDarkTheme?: (isDarkTheme: boolean) => void;
}

class App extends Component<AppProps> {
  render() {
    const { isDarkTheme = false, setIsDarkTheme = (val) => {} } = this.props;
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
              <header className="App-header">
                <div className="mlflow-logo">
                  <LinkV5 to={ExperimentTrackingRoutes.rootRoute} className="App-mlflow">
                    <img className="mlflow-logo" alt="MLflow" src={logo} />
                  </LinkV5>
                  <span className="mlflow-version">{Version}</span>
                </div>
                <div className="header-route-links">
                  <NavLinkV5
                    strict
                    to={ExperimentTrackingRoutes.rootRoute}
                    css={{ marginRight }}
                    activeStyle={classNames.activeNavLink}
                    isActive={isExperimentsActive}
                    className="header-nav-link"
                  >
                    <div className="experiments">
                      <span>Experiments</span>
                    </div>
                  </NavLinkV5>
                  <NavLinkV5
                    strict
                    to={ModelRegistryRoutes.modelListPageRoute}
                    css={{ marginRight }}
                    activeStyle={classNames.activeNavLink}
                    className="header-nav-link header-nav-link-models"
                  >
                    <div className="models">
                      <span>Models</span>
                    </div>
                  </NavLinkV5>
                </div>
                <div className="header-links">
                  <span css={{ marginRight }}>
                    <DarkThemeSwitch isDarkTheme={isDarkTheme} setIsDarkTheme={setIsDarkTheme} />
                  </span>
                  <a href="https://github.com/mlflow/mlflow" css={{ marginRight }}>
                    <div className="github">
                      <span>GitHub</span>
                    </div>
                  </a>
                  <a href={HomePageDocsUrl} css={{ marginRight }}>
                    <div className="docs">
                      <span>Docs</span>
                    </div>
                  </a>
                </div>
              </header>
            )}
            <AppErrorBoundary service="mlflow">
              <InteractionTracker>
                {/* The block below contains React Router v6 routes */}
                <Routes>
                  <Route path={ExperimentTrackingRoutePaths.compareExperimentsSearch} element={<HomePage />} />
                  <Route path={ExperimentTrackingRoutePaths.experimentPageSearch} element={<HomePage />} />
                  <Route path={ExperimentTrackingRoutePaths.experimentPage} element={<HomePage />} />
                  <Route path={ExperimentTrackingRoutePaths.rootRoute} element={<HomePage />} />
                  {/* If deep learning UI features are enabled, use more
                      versatile route (with backward compatibility) */}
                  {shouldEnableDeepLearningUI() ? (
                    <Route path={ExperimentTrackingRoutePaths.runPageWithTab} element={<RunPage />} />
                  ) : (
                    <>
                      <Route path={ExperimentTrackingRoutePaths.runPageWithArtifact} element={<RunPage />} />
                      <Route path={ExperimentTrackingRoutePaths.runPage} element={<RunPage />} />
                    </>
                  )}
                  <Route path={ExperimentTrackingRoutePaths.runPageDirect} element={<DirectRunPage />} />
                  <Route path={ExperimentTrackingRoutePaths.metricPage} element={<MetricPage />} />
                  <Route path={ExperimentTrackingRoutePaths.compareRuns} element={<CompareRunPage />} />
                  <Route path={ModelRegistryRoutePaths.modelListPage} element={<ModelListPage />} />
                  <Route path={ModelRegistryRoutePaths.modelVersionPage} element={<ModelVersionPage />} />
                  <Route path={ModelRegistryRoutePaths.modelPage} element={<ModelPage />} />
                  <Route path={ModelRegistryRoutePaths.modelSubpage} element={<ModelPage />} />
                  <Route path={ModelRegistryRoutePaths.modelSubpageRouteWithName} element={<ModelPage />} />
                  <Route
                    path={ModelRegistryRoutePaths.compareModelVersionsPage}
                    element={<CompareModelVersionsPage />}
                  />
                  <Route path="/*" element={<PageNotFoundView />} />
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
