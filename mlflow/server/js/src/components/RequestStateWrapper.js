import React, { Component } from 'react';
import './RequestStateWrapper.css';
import { connect } from 'react-redux';
import { getApis } from '../reducers/Reducers';
import PropTypes from 'prop-types';
import {Spinner} from "./Spinner";

export class RequestStateWrapper extends Component {
  static propTypes = {
    // Should this component render the child before all the requests are complete?
    shouldOptimisticallyRender: PropTypes.bool,
    requests: PropTypes.arrayOf(PropTypes.object).isRequired,
    // (isLoading: boolean, shouldRenderError: boolean, requests) => undefined | React Node.
    // This function is called when all requests are complete.
    // The function can choose to render an error view depending on the
    // type of errors received. If undefined is returned, then render the AppErrorBoundary view.
    children: PropTypes.oneOfType([PropTypes.func, PropTypes.element]),
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
    const { children, requests } = this.props;
    const { shouldRender, shouldRenderError } = this.state;
    if (shouldRender || this.props.shouldOptimisticallyRender) {
      if (typeof children === "function") {
        const child = children(!shouldRender, shouldRenderError, requests);
        if (child) {
          return child;
        }
        triggerError(requests);
      }
      if (shouldRenderError) {
        triggerError(requests);
      }
      return children;
    }
    return <Spinner/>;
  }
}

const triggerError = (requests) => {
  // This triggers the OOPS error boundary.
  console.error("ERROR", requests);
  throw Error("GOTO error boundary");
};

const mapStateToProps = (state, ownProps) => {
  return Object.assign({}, ownProps, {
    requests: getApis(ownProps.requestIds, state)
  });
};

export default connect(mapStateToProps)(RequestStateWrapper);
