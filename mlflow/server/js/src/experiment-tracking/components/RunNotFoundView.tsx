/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

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
