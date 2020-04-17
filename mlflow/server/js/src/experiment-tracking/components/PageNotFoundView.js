import React, { Component } from 'react';
import overflow from '../../common/static/404-overflow.svg';
import { Link } from 'react-router-dom';
import Routes from '../routes';

export class PageNotFoundView extends Component {
  render() {
    return (
      <div>
        <img
          className='center'
          alt='404 not found'
          style={{ height: '300px', marginTop: '80px' }}
          src={overflow}
        />
        <h1 className="center" style={{ paddingTop: '10px' }}>
          Page not found
        </h1>
        <h2 className="center" style={{ color: '#888' }}>
          Go back to <Link to={Routes.rootRoute}>the home page.</Link>
        </h2>
      </div>
    );
  }
}

