import React, { Component } from 'react';
import './App.css';
import logo from '../../common/static/home-logo.png';
import { HashRouter as Router, Route, Link, NavLink } from 'react-router-dom';
import { RunPage } from './RunPage';
import Routes from '../routes';
import { MetricPage } from './MetricPage';
import CompareRunPage from './CompareRunPage';
import AppErrorBoundary from '../../common/components/error-boundaries/AppErrorBoundary';
import { connect } from 'react-redux';
import { HomePage } from './HomePage';
import ErrorModal from '../../experiment-tracking/components/modals/ErrorModal';
import { PageNotFoundView } from './PageNotFoundView';
import { Switch } from 'react-router';
import {
  modelListPageRoute,
  modelPageRoute,
  modelSubpageRoute,
  modelVersionPageRoute,
  compareModelVersionsPageRoute,
} from '../../model-registry/routes';
import { ModelVersionPage } from '../../model-registry/components/ModelVersionPage';
import ModelListPage from '../../model-registry/components/ModelListPage';
import { ModelPage } from '../../model-registry/components/ModelPage';
import CompareModelVersionsPage from '../../model-registry/components/CompareModelVersionsPage';

const classNames = {
  activeNavLink: { borderBottom: '4px solid #43C9ED' },
};

class App extends Component {
  render() {
    return (
      <Router>
        <div style={{ height: '100vh' }}>
          <ErrorModal />
          {process.env.HIDE_HEADER === 'true' ? null : (
            <header className='App-header'>
              <div className='mlflow-logo'>
                <Link to={Routes.rootRoute} className='App-mlflow'>
                  <img className='mlflow-logo' alt='MLflow' src={logo} />
                </Link>
              </div>
              <div className='header-route-links'>
                <NavLink
                  strict
                  to={Routes.rootRoute}
                  activeStyle={classNames.activeNavLink}
                  isActive={(match, location) => match && !location.pathname.includes('models')}
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
                <a href={'https://mlflow.org/docs/latest/index.html'}>
                  <div className='docs'>
                    <span>Docs</span>
                  </div>
                </a>
              </div>
            </header>
          )}
          <AppErrorBoundary>
            <Switch>
              <Route exact path={Routes.rootRoute} component={HomePage} />
              <Route exact path={Routes.experimentPageRoute} component={HomePage} />
              <Route exact path={Routes.runPageWithArtifactSelectedRoute} component={RunPage} />
              <Route exact path={Routes.runPageRoute} component={RunPage} />
              <Route exact path={Routes.metricPageRoute} component={MetricPage} />
              <Route exact path={Routes.compareRunPageRoute} component={CompareRunPage} />
              <Route path={Routes.experimentPageSearchRoute} component={HomePage} />
              {/* TODO(Zangr) see if route component can be injected here */}
              <Route exact path={modelListPageRoute} component={ModelListPage} />
              <Route exact path={modelVersionPageRoute} component={ModelVersionPage} />
              <Route exact path={modelPageRoute} component={ModelPage} />
              <Route exact path={modelSubpageRoute} component={ModelPage} />
              <Route
                exact
                path={compareModelVersionsPageRoute}
                component={CompareModelVersionsPage}
              />
              <Route component={PageNotFoundView} />
            </Switch>
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
