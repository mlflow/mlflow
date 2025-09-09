/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import { connect } from 'react-redux';
import { getApis } from '../../experiment-tracking/reducers/Reducers';
import { Spinner } from './Spinner';
import { ErrorCodes } from '../constants';
import type { ErrorWrapper } from '../utils/ErrorWrapper';
import type { ReduxState } from '../../redux-types';

export const DEFAULT_ERROR_MESSAGE = 'A request error occurred.';

type RequestStateWrapperProps = {
  children?: React.ReactNode;
  customSpinner?: React.ReactNode;
  shouldOptimisticallyRender?: boolean;
  requests: any[];
  requestIds?: string[];
  requestIdsWith404sToIgnore?: string[];
  description?: any; // TODO: PropTypes.oneOf(Object.values(LoadingDescription))
  permissionDeniedView?: React.ReactNode;
  suppressErrorThrow?: boolean;
  customRequestErrorHandlerFn?: (
    failedRequests: {
      id: string;
      active?: boolean;
      error: Error | ErrorWrapper;
    }[],
  ) => void;
};

type RequestStateWrapperState = any;

export class RequestStateWrapper extends Component<RequestStateWrapperProps, RequestStateWrapperState> {
  static defaultProps = {
    requests: [],
    requestIdsWith404sToIgnore: [],
    shouldOptimisticallyRender: false,
  };

  state = {
    shouldRender: false,
    shouldRenderError: false,
  };

  static getErrorRequests(requests: any, requestIdsWith404sToIgnore: any) {
    return requests.filter((r: any) => {
      if (r.error !== undefined) {
        return !(
          requestIdsWith404sToIgnore &&
          requestIdsWith404sToIgnore.includes(r.id) &&
          r.error.getErrorCode() === ErrorCodes.RESOURCE_DOES_NOT_EXIST
        );
      }
      return false;
    });
  }

  static getDerivedStateFromProps(nextProps: any) {
    const shouldRender = nextProps.requests.length
      ? nextProps.requests.every((r: any) => r && r.active === false)
      : false;

    const requestErrors = RequestStateWrapper.getErrorRequests(
      nextProps.requests,
      nextProps.requestIdsWith404sToIgnore,
    );

    return {
      shouldRender,
      shouldRenderError: requestErrors.length > 0,
      requestErrors,
    };
  }

  getRenderedContent() {
    const { children, requests, customSpinner, permissionDeniedView, suppressErrorThrow, customRequestErrorHandlerFn } =
      this.props;
    // @ts-expect-error TS(2339): Property 'requestErrors' does not exist on type '{... Remove this comment to see the full error message
    const { shouldRender, shouldRenderError, requestErrors } = this.state;
    const permissionDeniedErrors = requestErrors.filter((failedRequest: any) => {
      return failedRequest.error.getErrorCode() === ErrorCodes.PERMISSION_DENIED;
    });

    if (typeof children === 'function') {
      return children(!shouldRender, shouldRenderError, requests, requestErrors);
    } else if (shouldRender || shouldRenderError || this.props.shouldOptimisticallyRender) {
      if (permissionDeniedErrors.length > 0 && permissionDeniedView) {
        return permissionDeniedView;
      }
      if (shouldRenderError && !suppressErrorThrow) {
        customRequestErrorHandlerFn ? customRequestErrorHandlerFn(requestErrors) : triggerError(requestErrors);
      }

      return children;
    }

    return customSpinner || <Spinner />;
  }

  render() {
    return this.getRenderedContent();
  }
}

export const triggerError = (requests: any) => {
  // This triggers the OOPS error boundary.
  // eslint-disable-next-line no-console -- TODO(FEINF-3587)
  console.error('ERROR', requests);
  throw Error(`${DEFAULT_ERROR_MESSAGE}: ${requests.error}`);
};

const mapStateToProps = (state: ReduxState, ownProps: Omit<RequestStateWrapperProps, 'requests'>) => ({
  requests: getApis(ownProps.requestIds, state),
});

export default connect(mapStateToProps)(RequestStateWrapper);
