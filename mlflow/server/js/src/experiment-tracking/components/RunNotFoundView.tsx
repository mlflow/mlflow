import React, { Component } from 'react';
import Routes from '../routes';
import { ErrorView } from '../../common/components/ErrorView';

type Props = {
  runId: string;
};

export class RunNotFoundView extends Component<Props> {
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
