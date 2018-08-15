import React, { Component } from 'react';
import './RequestStateWrapper.css';
import spinner from '../static/mlflow-spinner.png';
import { connect } from 'react-redux';
import { getApis } from '../reducers/Reducers';
import PropTypes from 'prop-types';

class RequestStateWrapper extends Component {
  static propTypes = {
    shouldOptimisticallyRender: PropTypes.bool,
    requests: PropTypes.arrayOf(PropTypes.object).isRequired,
    children: PropTypes.node,
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

  // eslint-disable-next-line no-unused-vars
  static getDerivedStateFromProps(nextProps, prevState) {
    const shouldRender = nextProps.requests.every((r) => {
      return r.active === false && r.error === undefined;
    });
    return {
      shouldRender,
      shouldRenderError: RequestStateWrapper.getErrorRequests(nextProps.requests).length > 0,
    };
  }

  render() {
    const { children } = this.props;
    if (this.state.shouldRender) {
      return <div>{children}</div>;
    }
    if (this.state.shouldRenderError) {
      const errorRequests = RequestStateWrapper.getErrorRequests(this.props.requests);
      const api = errorRequests.length > 0 ? errorRequests[0] : "";
      console.log("ERROR", api.error);
      return (
        <div className="RequestStateWrapper-error">
          {api.error.xhr.status}: {api.error.xhr.statusText}
        </div>
      );
    }
    if (this.props.shouldOptimisticallyRender) {
      return <div>{children}</div>;
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
