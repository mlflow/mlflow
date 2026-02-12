import { FormattedMessage } from 'react-intl';
import { EXPERIMENT_PAGE_FEEDBACK_URL, ExperimentPageTabName } from '@mlflow/mlflow/src/experiment-tracking/constants';
import { WorkflowType } from '@mlflow/mlflow/src/common/contexts/WorkflowTypeContext';
import {
  BeakerIcon,
  ChartLineIcon,
  DatabaseIcon,
  ForkHorizontalIcon,
  GavelIcon,
  ListIcon,
  ModelsIcon,
  PlusMinusSquareIcon,
  SpeechBubbleIcon,
  TextBoxIcon,
  UserGroupIcon,
} from '@databricks/design-system';
import { shouldEnableWorkflowBasedNavigation } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

export const getShareFeedbackOverflowMenuItem = () => {
  return {
    id: 'feedback',
    itemName: (
      <FormattedMessage
        defaultMessage="Send Feedback"
        description="Text for provide feedback button on experiment view page header"
      />
    ),
    href: EXPERIMENT_PAGE_FEEDBACK_URL,
  };
};

export const getTabDisplayIcon = (tabName: ExperimentPageTabName | undefined) => {
  if (!tabName || !shouldEnableWorkflowBasedNavigation()) {
    return <BeakerIcon />;
  }

  switch (tabName) {
    case ExperimentPageTabName.Overview:
      return <ChartLineIcon />;
    case ExperimentPageTabName.Runs:
      return <ListIcon />;
    case ExperimentPageTabName.Traces:
      return <ForkHorizontalIcon />;
    case ExperimentPageTabName.Models:
      return <ModelsIcon />;
    case ExperimentPageTabName.ChatSessions:
    case ExperimentPageTabName.SingleChatSession:
      return <SpeechBubbleIcon />;
    case ExperimentPageTabName.Datasets:
      return <DatabaseIcon />;
    case ExperimentPageTabName.EvaluationRuns:
      return <PlusMinusSquareIcon />;
    case ExperimentPageTabName.Judges:
      return <GavelIcon />;
    case ExperimentPageTabName.Prompts:
      return <TextBoxIcon />;
    case ExperimentPageTabName.LabelingSessions:
      return <UserGroupIcon />;
    case ExperimentPageTabName.LabelingSchemas:
      return <TextBoxIcon />;
    default:
      return <BeakerIcon />;
  }
};

export const getTabDisplayName = (tabName: ExperimentPageTabName, workflowType: WorkflowType) => {
  if (workflowType === WorkflowType.GENAI) {
    return getGenAITabDisplayName(tabName);
  }
  return getMLTabDisplayName(tabName);
};

export const getGenAITabDisplayName = (tabName: ExperimentPageTabName) => {
  switch (tabName) {
    case ExperimentPageTabName.Models:
      return (
        <FormattedMessage
          defaultMessage="Agent versions"
          description="Label for the agent versions tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Runs:
      return (
        <FormattedMessage
          defaultMessage="Training runs"
          description="Label for the training runs tab in the MLflow experiment navbar"
        />
      );
    default:
      return getMLTabDisplayName(tabName);
  }
};

export const getMLTabDisplayName = (tabName: ExperimentPageTabName) => {
  switch (tabName) {
    case ExperimentPageTabName.Overview:
      return (
        <FormattedMessage
          defaultMessage="Overview"
          description="Label for the overview tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Runs:
      return (
        <FormattedMessage
          defaultMessage="Runs"
          description="Label for the training runs tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Traces:
      return (
        <FormattedMessage
          defaultMessage="Traces"
          description="Label for the traces tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.SingleChatSession:
    case ExperimentPageTabName.ChatSessions:
      return (
        <FormattedMessage
          defaultMessage="Chat sessions"
          description="Label for the chat sessions tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Datasets:
      return (
        <FormattedMessage
          defaultMessage="Datasets"
          description="Label for the datasets tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.EvaluationRuns:
      return (
        <FormattedMessage
          defaultMessage="Evaluation runs"
          description="Label for the evaluation runs tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Judges:
      return (
        <FormattedMessage
          defaultMessage="Judges"
          description="Label for the judges tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.LabelingSessions:
      return (
        <FormattedMessage
          defaultMessage="Labeling sessions"
          description="Label for the labeling sessions tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.LabelingSchemas:
      return (
        <FormattedMessage
          defaultMessage="Labeling schemas"
          description="Label for the labeling schemas tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Prompts:
      return (
        <FormattedMessage
          defaultMessage="Prompts"
          description="Label for the prompts tab in the MLflow experiment navbar"
        />
      );
    case ExperimentPageTabName.Models:
      return (
        <FormattedMessage
          defaultMessage="Models"
          description="Label for the versions tab in the MLflow experiment navbar"
        />
      );
    default:
      return tabName;
  }
};
