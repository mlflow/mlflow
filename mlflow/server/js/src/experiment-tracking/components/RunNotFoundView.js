import React, { Component } from 'react';
import PropTypes from 'prop-types';
import overflow from '../../common/static/404-overflow.svg';
import Colors from '../styles/Colors';
import Routes from '../routes';
import { Link } from 'react-router-dom';
import { ErrorView } from '../../common/components/ErrorView';

export class RunNotFoundView extends Component {
  static propTypes = {
    runId: PropTypes.string.isRequired,
  };

  render() {
    return (
      <ErrorView
        statusCode={404}
        subMessage={`Run ID ${this.props.runId} does not exist`}
        fallbackHomePageReactRoute={Routes.rootRoute}
      />
    );
  }
}
