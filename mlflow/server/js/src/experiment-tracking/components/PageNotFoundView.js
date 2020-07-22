import React, { Component } from 'react';
import overflow from '../../common/static/404-overflow.svg';
import { Link } from 'react-router-dom';
import Routes from '../routes';
import { ErrorView } from '../../common/components/ErrorView';

export class PageNotFoundView extends Component {
  render() {
    return <ErrorView statusCode={404} fallbackHomePageReactRoute={Routes.rootRoute} />
  }
}
