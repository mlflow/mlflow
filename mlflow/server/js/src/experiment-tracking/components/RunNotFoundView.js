import React, { Component } from 'react';
import PropTypes from 'prop-types';
import overflow from '../../common/static/404-overflow.svg';
import Colors from '../styles/Colors';
import Routes from '../routes';
import { Link } from 'react-router-dom';


export class RunNotFoundView extends Component {
  static propTypes = {
    runId: PropTypes.string.isRequired,
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
          Page not found
        </h1>
        <h2 className="center" style={{ color: Colors.secondaryText }}>
          Run ID {this.props.runId} does not exist, go back to{' '}
          <Link to={Routes.rootRoute}>the home page.</Link>
        </h2>
      </div>
    );
  }
}
