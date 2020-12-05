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
            <header className='navbar navbar-expand flex-row App-header bd-navbar mb-4'>
              <Link to={Routes.rootRoute} className='navbar-brand'>
                <img className='mlflow-logo' alt='MLflow' src={logo} />
              </Link>
              <ul className='navbar-nav ml-3 mt-3'>
                <li className='nav-item'>
                  <NavLink
                    strict
                    to={Routes.rootRoute}
                    activeStyle={classNames.activeNavLink}
                    isActive={(match, location) => match && !location.pathname.includes('models')}
                    className='nav-link'
                  >
                    Experiments
                  </NavLink>
                </li>
                <li className='nav-item ml-1'>
                  <NavLink
                    strict
                    to={modelListPageRoute}
                    activeStyle={classNames.activeNavLink}
                    className='nav-link'
                  >
                    Models
                  </NavLink>
                </li>
              </ul>
              <ul className='navbar-nav ml-auto mr-5 mt-3'>
                <li className='nav-item'>
                  <a href={'https://github.com/mlflow/mlflow'} className='nav-link'>
                    GitHub
                  </a>
                </li>
                <li className='nav-item'>
                  <a href={'https://mlflow.org/docs/latest/index.html'} className='nav-link'>
                    Docs
                  </a>
                </li>
              </ul>
            </header>
          )}
          <AppErrorBoundary>
            <Switch>
              <Route exact path={Routes.rootRoute} component={HomePage} />
              <Route exact path={Routes.experimentPageRoute} component={HomePage} />
              <Route exact path={Routes.runPageRoute} component={RunPage} />
              <Route exact path={Routes.metricPageRoute} component={MetricPage} />
              <Route exact path={Routes.compareRunPageRoute} component={CompareRunPage} />
              <Route path={Routes.experimentPageSearchRoute} component={HomePage} />
              {/* TODO(Zangr) see if route component can be injected here */}
              <Route exact path={modelListPageRoute} component={ModelListPage} />
              <Route exact path={modelVersionPageRoute} component={ModelVersionPage} />
              <Route exact path={modelPageRoute} component={ModelPage} />
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
