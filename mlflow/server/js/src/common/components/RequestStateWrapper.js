import React, { Component } from 'react';
import './RequestStateWrapper.css';
import { connect } from 'react-redux';
import { getApis } from '../../experiment-tracking/reducers/Reducers';
import PropTypes from 'prop-types';
import { Spinner } from './Spinner';

export const DEFAULT_ERROR_MESSAGE = 'A request error occurred.';

export class RequestStateWrapper extends Component {
  static propTypes = {
    customSpinner: PropTypes.node,
    // Should this component render the child before all the requests are complete?
    shouldOptimisticallyRender: PropTypes.bool,
    requests: PropTypes.arrayOf(PropTypes.object).isRequired,
    // (isLoading: boolean, shouldRenderError: boolean, requests) => null | undefined | ReactNode.
    // This function is called when all requests are complete or some requests failed.
    // It's the function's responsibility to render a ReactNode or an error view depending on the
    // parameters passed in.
    children: PropTypes.oneOfType([
      PropTypes.func,
      PropTypes.element,
      PropTypes.arrayOf(PropTypes.element),
    ]),
  };

  static defaultProps = {
    requests: [],
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
    const shouldRender = nextProps.requests.length
      ? nextProps.requests.every((r) => r && r.active === false)
      : false;

    return {
      shouldRender,
      shouldRenderError: RequestStateWrapper.getErrorRequests(nextProps.requests).length > 0,
    };
  }

  render() {
    const { children, requests, customSpinner } = this.props;
    const { shouldRender, shouldRenderError } = this.state;
    if (typeof children === 'function') {
      return children(!shouldRender, shouldRenderError, requests);
    } else if (shouldRender || shouldRenderError || this.props.shouldOptimisticallyRender) {
      if (shouldRenderError) {
        triggerError(requests);
      }
      return children;
    }

    return customSpinner || <Spinner />;
  }
}

export const triggerError = (requests) => {
  // This triggers the OOPS error boundary.
  console.error('ERROR', requests);
  throw Error(DEFAULT_ERROR_MESSAGE);
};

const mapStateToProps = (state, ownProps) => ({
  requests: getApis(ownProps.requestIds, state),
});

export default connect(mapStateToProps)(RequestStateWrapper);
