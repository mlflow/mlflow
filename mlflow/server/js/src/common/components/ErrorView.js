import React, { Component } from 'react';
import PropTypes from 'prop-types';
import errorDefaultImg from '../static/default-error.svg';
import error404Img from '../static/404-overflow.svg';
import Colors from '../../experiment-tracking/styles/Colors';
import Routes from '../../experiment-tracking/routes';
import { Link } from 'react-router-dom';

const altMessages = {
  400: '400 Bad Request',
  401: '401 Unauthorized',
  404: '404 Not Found',
  409: '409 Conflict',
  500: '500 Internal Server Error',
  502: '502 Bad Gateway',
  503: '503 Service Unavailable',
};

function ErrorImage(props) {
  const { statusCode } = props;
  const alt = altMessages[statusCode] || statusCode.toString();

  switch (props.statusCode) {
    case 404:
      return (
        <img
          className='center'
          alt={alt}
          style={{ height: '300px', marginTop: '80px' }}
          src={error404Img}
        />
      );
    default:
      return (
        <img
          className='center'
          alt={alt}
          src={errorDefaultImg}
          style={{
            margin: '12% auto 60px',
            display: 'block',
          }}
        />
      );
  }
}

ErrorImage.propTypes = { statusCode: PropTypes.number.isRequired };

export class ErrorView extends Component {
  static propTypes = {
    statusCode: PropTypes.number.isRequired,
    subMessage: PropTypes.string,
    fallbackHomePageReactRoute: PropTypes.string,
  };

  static centerMessages = {
    400: 'Bad Request',
    404: 'Page Not Found',
  };

  render() {
    const { statusCode, subMessage, fallbackHomePageReactRoute } = this.props;
    const centerMessage = ErrorView.centerMessages[statusCode] || 'HTTP Request Error';

    return (
      <div>
        <ErrorImage statusCode={statusCode} />
        <h1 className='center' style={{ paddingTop: '10px' }}>
          {centerMessage}
        </h1>
        <h2 className='center' style={{ color: Colors.secondaryText }}>
          {subMessage ? subMessage + ', go back to ' : 'Go back to '}
          <Link to={fallbackHomePageReactRoute || Routes.rootRoute}>the home page.</Link>
        </h2>
      </div>
    );
  }
}
