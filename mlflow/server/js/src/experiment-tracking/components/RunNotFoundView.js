import React, { Component } from 'react';
import PropTypes from 'prop-types';
import Routes from '../routes';
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
