import React from 'react';
import PropTypes from 'prop-types';
import Utils from '../../utils/Utils';
import { css } from 'emotion';

export class SectionErrorBoundary extends React.Component {
  static propTypes = {
    children: PropTypes.node,
    showServerError: PropTypes.bool,
  };

  state = { error: null };

  componentDidCatch(error, errorInfo) {
    this.setState({ error });
    console.error(error, errorInfo);
  }

  renderErrorMessage(error) {
    return this.props.showServerError ? <div>Error message: {error.message}</div> : '';
  }

  render() {
    const { children } = this.props;
    const { error } = this.state;
    if (error) {
      return (
        <div>
          <p>
            <i className={`fa fa-exclamation-triangle icon-fail ${classNames.wrapper}`} />
            <span> Something went wrong with this section. </span>
            <span>If this error persists, please report an issue </span>
            <a href={Utils.getSupportPageUrl()} target='_blank'>
              here
            </a>
            .{this.renderErrorMessage(error)}
          </p>
        </div>
      );
    }

    return children;
  }
}

const classNames = {
  wrapper: css({
    marginLeft: -2, // to align the failure icon with the collapsable section caret toggle
  }),
};
