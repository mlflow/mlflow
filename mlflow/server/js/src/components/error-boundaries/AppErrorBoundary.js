import React, { Component } from 'react';
import './AppErrorBoundary.css';
import defaultErrorImg from '../../static/default-error.svg';
import PropTypes from 'prop-types';

class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static propTypes = {
    children: PropTypes.node
  };

  // eslint-disable-next-line no-unused-vars
  componentDidCatch(error, info) {
    this.setState({ hasError: true });
    console.error(error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div>
          <img className="error-image" alt="Error" src={defaultErrorImg}/>
          <h1 className={"center"}>Something went wrong</h1>
          <h4 className={"center"}>
            If this error persists, please report an issue on {' '}
            <a href="https://github.com/mlflow/mlflow/issues">our GitHub page</a>.
          </h4>
        </div>
      );
    }
    return this.props.children;
  }
}

export default AppErrorBoundary;
