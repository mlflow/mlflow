import React, { Component } from 'react';
import './RequestStateWrapper.css';
import spinner from '../static/mlflow-spinner.png';
import { connect } from 'react-redux';
import { getApis } from '../reducers/Reducers';
import PropTypes from 'prop-types';

export class RequestStateWrapper extends Component {
  static propTypes = {
    shouldOptimisticallyRender: PropTypes.bool,
    requests: PropTypes.arrayOf(PropTypes.object).isRequired,
    children: PropTypes.node.isRequired,
    // (requests) => undefined | React Node.
    // This function is called when all requests are complete and when one or more of them is
    // in the error state. The function can choose to render an error view depending on the
    // type of errors received. If undefined is returned, then render the AppErrorBoundary view.
    errorRenderFunc: PropTypes.func,
  };

  static defaultProps = {
    shouldOptimisticallyRender: false,
  };

  state = {
    shouldRender: false,
    shouldRenderError: false,
  };

  static getErrorRequests(requests) {
    return requests.filter((r) => {
      return r.error !== undefined;
    });
  }

  static getDerivedStateFromProps(nextProps) {
    const shouldRender = nextProps.requests.every((r) => {
      return r.active === false;
    });
    return {
      shouldRender,
      shouldRenderError: RequestStateWrapper.getErrorRequests(nextProps.requests).length > 0,
    };
  }

  render() {
    const { children, errorRenderFunc, requests } = this.props;
    const { shouldRender, shouldRenderError } = this.state;
    if (shouldRender) {
      if (shouldRenderError) {
        if (errorRenderFunc) {
          const result = errorRenderFunc(this.props.requests);
          if (result) {
            return result;
          }
        }
        // This triggers the OOPS error boundary.
        console.error("ERROR", requests);
        throw Error("GOTO error boundary");
      } else {
        return children;
      }
    }
    if (this.props.shouldOptimisticallyRender) {
      return children;
    }
    return (
      <div className="RequestStateWrapper-spinner">
        <img alt="Page loading..." src={spinner}/>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => {
  return Object.assign({}, ownProps, {
    requests: getApis(ownProps.requestIds, state)
  });
};

export default connect(mapStateToProps)(RequestStateWrapper);
