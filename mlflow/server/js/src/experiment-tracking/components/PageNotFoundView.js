import React, { Component } from 'react';
import Routes from '../routes';
import { ErrorView } from '../../common/components/ErrorView';

export class PageNotFoundView extends Component {
  render() {
    return <ErrorView statusCode={404} fallbackHomePageReactRoute={Routes.rootRoute} />;
  }
}
