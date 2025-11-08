import { Component } from 'react';
import errorDefaultImg from '../static/default-error.svg';
import error404Img from '../static/404-overflow.svg';
import Routes from '../../experiment-tracking/routes';
import { Link } from '../utils/RoutingUtils';
import { FormattedMessage } from 'react-intl';
import type { DesignSystemHocProps } from '@databricks/design-system';
import { WithDesignSystemThemeHoc } from '@databricks/design-system';

const altMessages: Record<number, string> = {
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
  const alt = altMessages[statusCode] || statusCode.toString();

  switch (props.statusCode) {
    case 404:
      return (
        <img className="mlflow-center" alt={alt} style={{ height: '300px', marginTop: '80px' }} src={error404Img} />
      );
    default:
      return (
        <img
          className="mlflow-center"
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

type ErrorViewImplProps = DesignSystemHocProps & {
  statusCode: number;
  subMessage?: string;
  fallbackHomePageReactRoute?: string;
};

class ErrorViewImpl extends Component<ErrorViewImplProps> {
  static centerMessages: Record<number, string> = {
    400: 'Bad Request',
    404: 'Page Not Found',
    409: 'Resource Conflict',
  };

  renderErrorMessage(subMessage?: string, fallbackHomePageReactRoute?: string) {
    if (subMessage) {
      return (
        <FormattedMessage
          defaultMessage="{subMessage}, go back to <link>the home page.</link>"
          description="Default error message for error views in MLflow"
          values={{
            link: (chunks) => (
              <Link data-testid="error-view-link" to={fallbackHomePageReactRoute || Routes.rootRoute}>
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
            link: (chunks) => (
              <Link data-testid="error-view-link" to={fallbackHomePageReactRoute || Routes.rootRoute}>
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
    const centerMessage = ErrorViewImpl.centerMessages[statusCode] || 'HTTP Request Error';

    return (
      <div className="mlflow-center">
        <ErrorImage statusCode={statusCode} />
        <h1 style={{ paddingTop: '10px' }}>{centerMessage}</h1>
        <h2 style={{ color: designSystemThemeApi.theme.colors.textSecondary }}>
          {this.renderErrorMessage(subMessage, fallbackHomePageReactRoute)}
        </h2>
      </div>
    );
  }
}

export const ErrorView = WithDesignSystemThemeHoc(ErrorViewImpl);
