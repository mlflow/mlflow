import React, { Component } from 'react';
import PropTypes from 'prop-types';
import error404Img from '../../static/404-overflow.svg';
import Colors from '../../styles/Colors';
import Routes from '../../Routes';
import { Link } from 'react-router-dom';


export class Error404View extends Component {
  static propTypes = {
    resourceName: PropTypes.string.isRequired,
    fallbackHomePageReactRoute: PropTypes.string,
  };

  render() {
    const { resourceName, fallbackHomePageReactRoute } = this.props;
    return (
      <div>
        <img
          className='center'
          alt="404 Not Found"
          style={{ height: '300px', marginTop: '80px' }}
          src={error404Img}
        />
        <h1 className="center" style={{ paddingTop: '10px' }}>
          Page not found
        </h1>
        <h2 className="center" style={{ color: Colors.secondaryText }}>
          {resourceName} does not exist, go back to{' '}
          <Link to={fallbackHomePageReactRoute || Routes.rootRoute}>the home page.</Link>
        </h2>
      </div>
    );
  }
}
