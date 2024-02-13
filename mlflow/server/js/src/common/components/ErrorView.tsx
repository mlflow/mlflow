/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React, { Component } from 'react';
import errorDefaultImg from '../static/default-error.svg';
import error404Img from '../static/404-overflow.svg';
import Routes from '../../experiment-tracking/routes';
import { Link } from '../../common/utils/RoutingUtils';
import { FormattedMessage } from 'react-intl';
import { WithDesignSystemThemeHoc } from '@databricks/design-system';

const altMessages = {
  400: '400 Bad Request',
  401: '401 Unauthorized',
  404: '404 Not Found',
  409: '409 Conflict',
  500: '500 Internal Server Error',
  502: '502 Bad Gateway',
  503: '503 Service Unavailable',
};

type ErrorImageProps = {
  statusCode: number;
};

function ErrorImage(props: ErrorImageProps) {
  const { statusCode } = props;
  // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
  const alt = altMessages[statusCode] || statusCode.toString();

  switch (props.statusCode) {
    case 404:
      return <img className="center" alt={alt} style={{ height: '300px', marginTop: '80px' }} src={error404Img} />;
    default:
      return (
        <img
          className="center"
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

type ErrorViewImplProps = {
  statusCode: number;
  subMessage?: string;
  fallbackHomePageReactRoute?: string;
  designSystemThemeApi?: any;
};

export class ErrorViewImpl extends Component<ErrorViewImplProps> {
  static centerMessages = {
    400: 'Bad Request',
    404: 'Page Not Found',
    409: 'Resource Conflict',
  };

  renderErrorMessage(subMessage: any, fallbackHomePageReactRoute: any) {
    if (subMessage) {
      return (
        <FormattedMessage
          defaultMessage="{subMessage}, go back to <link>the home page.</link>"
          description="Default error message for error views in MLflow"
          values={{
            link: (chunks: any) => (
              <Link data-test-id="error-view-link" to={fallbackHomePageReactRoute || Routes.rootRoute}>
                {chunks}
              </Link>
            ),
            subMessage: subMessage,
          }}
        />
      );
    } else {
      return (
        <FormattedMessage
          defaultMessage="Go back to <link>the home page.</link>"
          description="Default error message for error views in MLflow"
          values={{
            link: (chunks: any) => (
              <Link data-test-id="error-view-link" to={fallbackHomePageReactRoute || Routes.rootRoute}>
                {chunks}
              </Link>
            ),
          }}
        />
      );
    }
  }

  render() {
    const { statusCode, subMessage, fallbackHomePageReactRoute, designSystemThemeApi } = this.props;
    const { theme } = designSystemThemeApi;
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    const centerMessage = ErrorViewImpl.centerMessages[statusCode] || 'HTTP Request Error';

    return (
      <div className="center">
        <ErrorImage statusCode={statusCode} />
        <h1 style={{ paddingTop: '10px' }}>{centerMessage}</h1>
        <h2 style={{ color: theme.colors.textSecondary }}>
          {this.renderErrorMessage(subMessage, fallbackHomePageReactRoute)}
        </h2>
      </div>
    );
  }
}

// @ts-expect-error TS(2345): Argument of type 'typeof ErrorViewImpl' is not ass... Remove this comment to see the full error message
export const ErrorView = WithDesignSystemThemeHoc(ErrorViewImpl);
