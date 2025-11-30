import {
  CheckCircleIcon,
  ClockIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
  XCircleIcon,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { LoggedModelStatusProtoEnum, type LoggedModelProto } from '../../types';

const LoggedModelStatusIcon = ({ status }: { status: LoggedModelStatusProtoEnum }) => {
  if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_READY) {
    return <CheckCircleIcon color="success" />;
  }

  if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_UPLOAD_FAILED) {
    return <XCircleIcon color="danger" />;
  }

  if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_PENDING) {
    return <ClockIcon color="warning" />;
  }

  return null;
};

export const ExperimentLoggedModelStatusIndicator = ({ data }: { data: LoggedModelProto }) => {
  const { theme } = useDesignSystemTheme();
  const status = data.info?.status ?? LoggedModelStatusProtoEnum.LOGGED_MODEL_STATUS_UNSPECIFIED;

  const getTagColor = () => {
    if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_READY) {
      return theme.isDarkMode ? theme.colors.green800 : theme.colors.green100;
    }
    if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_UPLOAD_FAILED) {
      return theme.isDarkMode ? theme.colors.red800 : theme.colors.red100;
    }
    if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_PENDING) {
      return theme.isDarkMode ? theme.colors.yellow800 : theme.colors.yellow100;
    }

    return undefined;
  };

  const getStatusLabel = () => {
    if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_READY) {
      return (
        <Typography.Text color="success">
          <FormattedMessage defaultMessage="Ready" description="Label for ready state of a experiment logged model" />
        </Typography.Text>
      );
    }

    if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_UPLOAD_FAILED) {
      return (
        <Typography.Text color="error">
          <FormattedMessage
            defaultMessage="Failed"
            description="Label for upload failed state of a experiment logged model"
          />
        </Typography.Text>
      );
    }
    if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_PENDING) {
      return (
        <Typography.Text color="warning">
          <FormattedMessage
            defaultMessage="Pending"
            description="Label for pending state of a experiment logged model"
          />
        </Typography.Text>
      );
    }

    return status;
  };

  if (status === LoggedModelStatusProtoEnum.LOGGED_MODEL_STATUS_UNSPECIFIED) {
    return null;
  }

  return (
    <Tag componentId="mlflow.logged_model.status" css={{ backgroundColor: getTagColor() }}>
      {status && <LoggedModelStatusIcon status={status} />}{' '}
      <Typography.Text css={{ marginLeft: theme.spacing.sm }}>{getStatusLabel()}</Typography.Text>
    </Tag>
  );
};
