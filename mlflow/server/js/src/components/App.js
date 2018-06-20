import React, { Component } from 'react';
import './App.css';
import logo from '../static/home-logo.png';
import ExperimentPage from './ExperimentPage';
import {
  BrowserRouter as Router,
  Route,
  Link,
} from 'react-router-dom';
import RunPage from './RunPage';
import Routes from '../Routes';
import MetricPage from './MetricPage';
import CompareRunPage from './CompareRunPage';
import AppErrorBoundary from './error-boundaries/AppErrorBoundary';
import { connect } from 'react-redux';
import HomePage from './HomePage';

class App extends Component {
  render() {
    return (
      <Router>
        <div>
          <header className="App-header">
            <div className="mlflow-logo">
              <Link
                to={Routes.rootRoute}
                className="App-mlflow"
              >
                <img className="mlflow-logo" alt="MLflow" src={logo}/>
              </Link>
            </div>
            <div className="header-links">
              <a href={'https://github.com/databricks/mlflow'}>
                <div className="github">
                  <span>GitHub</span>
                </div>
              </a>
              <a href={'https://mlflow.org/docs/latest/index.html'}>
                <div className="docs">
                  <span>Docs</span>
                </div>
              </a>
            </div>
          </header>
          <AppErrorBoundary>
            { /* Since the experiment list side bar goes outside of the 80% width put outside of div */ }
            <Route exact path={Routes.rootRoute} component={HomePage}/>
            <Route exact path={Routes.experimentPageRoute} component={HomePage}/>
            { /* App-content ensures 80% width */ }
            <div className="App-content">
                <Route exact path={Routes.runPageRoute} component={RunPage}/>
                <Route exact path={Routes.metricPageRoute} component={MetricPage}/>
                <Route exact path={Routes.compareRunPageRoute} component={CompareRunPage}/>
            </div>
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
