import React, { Component } from 'react';
import { connect } from 'react-redux';
import { HashRouter as Router, Link, NavLink, Route, Switch } from 'react-router-dom';
import AppErrorBoundary from '../../common/components/error-boundaries/AppErrorBoundary';
import { HomePageDocsUrl, Version } from '../../common/constants';
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
import Routes from '../routes';
import './App.css';
import CompareRunPage from './CompareRunPage';
import { HomePage } from './HomePage';
import { MetricPage } from './MetricPage';
import { PageNotFoundView } from './PageNotFoundView';
import { RunPage } from './RunPage';

const isExperimentsActive = (match, location) => {
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
const InteractionTracker = ({ children }) => children;

class App extends Component {
  render() {
    return (
      <Router
        basename={mlflowHashRouting ? '/mlflow' : undefined}
        hashType={mlflowHashRouting ? 'noslash' : undefined}
      >
        <div style={{ height: '100vh' }}>
          <ErrorModal />
          {process.env.HIDE_HEADER === 'true' ? null : (
            <header className='App-header'>
              <div className='mlflow-logo'>
                <Link to={Routes.rootRoute} className='App-mlflow'>
                  <img className='mlflow-logo' alt='MLflow' src={logo} />
                </Link>
                <span className={'mlflow-version'}>{Version}</span>
              </div>
              <div className='header-route-links'>
                <NavLink
                  strict
                  to={Routes.rootRoute}
                  activeStyle={classNames.activeNavLink}
                  isActive={isExperimentsActive}
                  className='header-nav-link'
                >
                  <div className='experiments'>
                    <span>Experiments</span>
                  </div>
                </NavLink>
                <NavLink
                  strict
                  to={modelListPageRoute}
                  activeStyle={classNames.activeNavLink}
                  className='header-nav-link header-nav-link-models'
                >
                  <div className='models'>
                    <span>Models</span>
                  </div>
                </NavLink>
              </div>
              <div className='header-links'>
                <a href={'https://github.com/mlflow/mlflow'}>
                  <div className='github'>
                    <span>GitHub</span>
                  </div>
                </a>
                <a href={HomePageDocsUrl}>
                  <div className='docs'>
                    <span>Docs</span>
                  </div>
                </a>
              </div>
            </header>
          )}
          <AppErrorBoundary service='mlflow'>
            <InteractionTracker>
              <Switch>
                <Route exact path={Routes.rootRoute} component={HomePage} />
                <Route exact path={Routes.experimentPageRoute} component={HomePage} />
                <Route exact path={Routes.runPageWithArtifactSelectedRoute} component={RunPage} />
                <Route exact path={Routes.runPageRoute} component={RunPage} />
                <Route exact path={Routes.metricPageRoute} component={MetricPage} />
                <Route exact path={Routes.compareRunPageRoute} component={CompareRunPage} />
                <Route exact path={Routes.compareExperimentsSearchPageRoute} component={HomePage} />
                <Route path={Routes.experimentPageSearchRoute} component={HomePage} />
                {/* TODO(Zangr) see if route component can be injected here */}
                <Route exact path={modelListPageRoute} component={ModelListPage} />
                <Route exact path={modelVersionPageRoute} component={ModelVersionPage} />
                <Route exact path={modelPageRoute} component={ModelPage} />
                <Route exact path={modelSubpageRoute} component={ModelPage} />
                <Route exact path={modelSubpageRouteWithName} component={ModelPage} />
                <Route
                  exact
                  path={compareModelVersionsPageRoute}
                  component={CompareModelVersionsPage}
                />
                <Route component={PageNotFoundView} />
              </Switch>
            </InteractionTracker>
          </AppErrorBoundary>
        </div>
      </Router>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    experiments: Object.values(state.entities.experimentsById),
  };
};

export default connect(mapStateToProps)(App);
