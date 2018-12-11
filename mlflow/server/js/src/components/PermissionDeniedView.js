import React, { Component } from 'react';
import PropTypes from 'prop-types';
import overflow from '../static/404-overflow.svg';
import Colors from '../styles/Colors';

export default class PermissionDeniedView extends Component {
  static propTypes = {
    errorMessage: PropTypes.string.isRequired,
  };

  render() {
    return (
      <div>
        <img
          className='center'
          alt="404 Not Found"
          style={{ height: '300px', marginTop: '80px' }}
          src={overflow}
        />
        <h1 className="center" style={{ paddingTop: '10px' }}>
          Permission Denied
        </h1>
        <h2 className="center" style={{ color: Colors.secondaryText }}>
          {this.props.errorMessage || 'The current user does not have permission to view this page.'}
        </h2>
      </div>
    );
  }
}
