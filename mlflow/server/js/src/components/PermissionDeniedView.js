import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Colors from '../styles/Colors';

export default class PermissionDeniedView extends Component {
  static propTypes = {
    errorMessage: PropTypes.string.isRequired,
  };

  render() {
    const defaultMessage = 'The current user does not have permission to view this page.';
    return (
      <div>
        <h1 className="center" style={{ paddingTop: '100px' }}>
          Permission Denied
        </h1>
        <h2 className="center" style={{ color: Colors.secondaryText }}>
          {this.props.errorMessage || defaultMessage}
        </h2>
      </div>
    );
  }
}
