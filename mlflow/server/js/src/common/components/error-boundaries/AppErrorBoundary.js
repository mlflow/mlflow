import React, { Component } from 'react';
import './AppErrorBoundary.css';
import defaultErrorImg from '../../static/default-error.svg';
import PropTypes from 'prop-types';
import Utils from '../../utils/Utils';
import { withNotifications, Accordion, Typography } from '@databricks/design-system';

class AppErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static propTypes = {
    children: PropTypes.node,
    notificationAPI: PropTypes.object,
    notificationContextHolder: PropTypes.node,
  };

  componentDidMount() {
    // Register this component's notifications API (corresponding to locally mounted
    // notification context) in the global Utils object.
    Utils.registerNotificationsApi(this.props.notificationAPI);
  }

  componentDidCatch(error, errorInfo) {
    this.setState({ hasError: true, error });
    console.error(error, errorInfo);
  }

  render() {
    const { error } = this.state;
    return (
      <>
        {this.state.hasError ? (
          <div>
            <img className='error-image' alt='Error' src={defaultErrorImg} />
            <h1 className={'center'}>Something went wrong: {error.toString()}</h1>
            <Accordion>
              <Accordion.Panel header='Show details'>
                <Typography css={{ fontSize: 10, whiteSpace: 'pre-wrap' }}>
                  {error.stack.toString()}
                </Typography>
              </Accordion.Panel>
            </Accordion>
            <h4 className={'center'}>
              If this error persists, please report an issue {/* Reported during ESLint upgrade */}
              {/* eslint-disable-next-line react/jsx-no-target-blank */}
              <a href={Utils.getSupportPageUrl()} target='_blank'>
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

export default withNotifications(AppErrorBoundary);
