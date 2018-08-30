import React, { Component } from 'react';
import './AppErrorBoundary.css';
import niagara from '../../static/niagara.jpg';
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
          <h1 className={"center"}>Oops! Something went wrong.</h1>
          <h4 className={"center"}>
            If this error persists, please report an issue on {' '}
            <a href="https://github.com/mlflow/mlflow/issues">our GitHub page</a>.
          </h4>
          <img className="niagara" alt="The Niagara falls." src={niagara}/>
        </div>
      );
    }
    return this.props.children;
  }
}

export default AppErrorBoundary;
