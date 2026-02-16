/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import './AppErrorBoundary.css';
import defaultErrorImg from '../../static/default-error.svg';
import Utils from '../../utils/Utils';
import { withNotifications } from '@databricks/design-system';

type Props = {
  service?: string;
  notificationAPI?: any;
  notificationContextHolder?: React.ReactNode;
};

type State = any;

class AppErrorBoundary extends Component<React.PropsWithChildren<Props>, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidMount() {
    // Register this component's notifications API (corresponding to locally mounted
    // notification context) in the global Utils object.
    Utils.registerNotificationsApi(this.props.notificationAPI);
  }

  componentDidCatch(error: any, errorInfo: any) {
    this.setState({ hasError: true });
    // eslint-disable-next-line no-console -- TODO(FEINF-3587)
    console.error(error, errorInfo);
  }

  render() {
    return (
      <>
        {/* @ts-expect-error TS(4111): Property 'hasError' comes from an index signature,... Remove this comment to see the full error message */}
        {this.state.hasError ? (
          <div>
            <img className="mlflow-error-image" alt="Error" src={defaultErrorImg} />
            <h1 className="mlflow-center">Something went wrong</h1>
            <h4 className="mlflow-center">
              If this error persists, please report an issue {/* Reported during ESLint upgrade */}
              {/* eslint-disable-next-line react/jsx-no-target-blank */}
              <a href={Utils.getSupportPageUrl()} target="_blank">
                here
              </a>
              .
            </h4>
          </div>
        ) : (
          this.props.children
        )}
        {this.props.notificationContextHolder}
      </>
    );
  }
}

// @ts-expect-error TS(2345): Argument of type 'typeof AppErrorBoundary' is not ... Remove this comment to see the full error message
export default withNotifications(AppErrorBoundary);
