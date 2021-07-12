import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Colors from '../styles/Colors';
import permissionDeniedLock from '../../common/static/permission-denied-lock.svg';

export class PermissionDeniedView extends Component {
  static propTypes = {
    errorMessage: PropTypes.string.isRequired,
  };

  render() {
    const defaultMessage = 'The current user does not have permission to view this page.';
    return (
      <div>
        <img
          className='center'
          style={{ height: 300, marginTop: 80 }}
          src={permissionDeniedLock}
          alt='permission denied'
        />
        <h1 className='center' style={{ paddingTop: 10 }}>
          Permission Denied
        </h1>
        <h2 className='center' style={{ color: Colors.secondaryText }}>
          {this.props.errorMessage || defaultMessage}
        </h2>
      </div>
    );
  }
}
