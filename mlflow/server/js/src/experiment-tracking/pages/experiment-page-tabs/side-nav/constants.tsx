import React from 'react';
import { ExperimentKind } from '../../../constants';
import { ExperimentPageTabName } from '../../../constants';
import {
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
import { FormattedMessage } from 'react-intl';
import { enableScorersUI } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

export const FULL_WIDTH_CLASS_NAME = 'mlflow-experiment-page-side-nav-full';
export const COLLAPSED_CLASS_NAME = 'mlflow-experiment-page-side-nav-collapsed';

export type ExperimentPageSideNavItem = {
  componentId: string;
  label: React.ReactNode;
  icon: React.ReactNode;
  tabName: ExperimentPageTabName;
};

export type ExperimentPageSideNavConfig = {
  [section in ExperimentPageSideNavSectionKey]?: ExperimentPageSideNavItem[];
};

export type ExperimentPageSideNavSectionKey = 'top-level' | 'observability' | 'evaluation' | 'prompts-versions';

const ExperimentPageSideNavGenAIConfig = {
  observability: [
    {
      label: (
        <FormattedMessage
          defaultMessage="Traces"
          description="Label for the traces tab in the MLflow experiment navbar"
        />
      ),
      icon: <ForkHorizontalIcon />,
      tabName: ExperimentPageTabName.Traces,
      componentId: 'mlflow.experiment-side-nav.genai.traces',
    },
  ],
  evaluation: [
    {
      label: (
        <FormattedMessage
          defaultMessage="Datasets"
          description="Label for the datasets tab in the MLflow experiment navbar"
        />
      ),
      icon: <DatabaseIcon />,
      tabName: ExperimentPageTabName.Datasets,
      componentId: 'mlflow.experiment-side-nav.genai.datasets',
    },
    {
      label: (
        <FormattedMessage
          defaultMessage="Evaluation runs"
          description="Label for the evaluation runs tab in the MLflow experiment navbar"
        />
      ),
      icon: <PlusMinusSquareIcon />,
      tabName: ExperimentPageTabName.EvaluationRuns,
      componentId: 'mlflow.experiment-side-nav.genai.evaluation-runs',
    },
  ],
  'prompts-versions': [
    {
      label: (
        <FormattedMessage
          defaultMessage="Prompts"
          description="Label for the prompts tab in the MLflow experiment navbar"
        />
      ),
      icon: <TextBoxIcon />,
      tabName: ExperimentPageTabName.Prompts,
      componentId: 'mlflow.experiment-side-nav.genai.prompts',
    },
    {
      label: (
        <FormattedMessage
          defaultMessage="Agent versions"
          description="Label for the agent versions tab in the MLflow experiment navbar"
        />
      ),
      icon: <ModelsIcon />,
      tabName: ExperimentPageTabName.Models,
      componentId: 'mlflow.experiment-side-nav.genai.agent-versions',
    },
  ],
};

const ExperimentPageSideNavCustomModelConfig = {
  'top-level': [
    {
      label: (
        <FormattedMessage defaultMessage="Runs" description="Label for the runs tab in the MLflow experiment navbar" />
      ),
      icon: <ListIcon />,
      tabName: ExperimentPageTabName.Runs,
      componentId: 'mlflow.experiment-side-nav.classic-ml.runs',
    },
    {
      label: (
        <FormattedMessage
          defaultMessage="Models"
          description="Label for the Models tab in the MLflow experiment navbar"
        />
      ),
      icon: <ModelsIcon />,
      tabName: ExperimentPageTabName.Models,
      componentId: 'mlflow.experiment-side-nav.classic-ml.models',
    },
    {
      label: (
        <FormattedMessage
          defaultMessage="Traces"
          description="Label for the traces tab in the MLflow experiment navbar"
        />
      ),
      icon: <ForkHorizontalIcon />,
      tabName: ExperimentPageTabName.Traces,
      componentId: 'mlflow.experiment-side-nav.classic-ml.traces',
    },
  ],
};

export const getExperimentPageSideNavSectionLabel = (
  section: ExperimentPageSideNavSectionKey,
): React.ReactNode | undefined => {
  switch (section) {
    case 'observability':
      return (
        <FormattedMessage
          defaultMessage="Observability"
          description="Label for the observability section in the MLflow experiment navbar"
        />
      );
    case 'evaluation':
      return (
        <FormattedMessage
          defaultMessage="Evaluation"
          description="Label for the evaluation section in the MLflow experiment navbar"
        />
      );
    case 'prompts-versions':
      return (
        <FormattedMessage
          defaultMessage="Prompts & versions"
          description="Label for the versions section in the MLflow experiment navbar"
        />
      );
    default:
      // no label for top-level section
      return undefined;
  }
};

export const useExperimentPageSideNavConfig = ({
  experimentKind,
  hasTrainingRuns = false,
}: {
  experimentKind: ExperimentKind;
  hasTrainingRuns?: boolean;
}): ExperimentPageSideNavConfig => {
  if (
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT ||
    experimentKind === ExperimentKind.GENAI_DEVELOPMENT_INFERRED
  ) {
    return {
      ...(hasTrainingRuns
        ? {
            // append training runs to top-level if they exist
            'top-level': [
              {
                label: (
                  <FormattedMessage
                    defaultMessage="Training runs"
                    description="Label for the training runs tab in the MLflow experiment navbar"
                  />
                ),
                icon: <ListIcon />,
                tabName: ExperimentPageTabName.Runs,
                componentId: 'mlflow.experiment-side-nav.genai.training-runs',
              },
            ],
          }
        : {
            'top-level': [],
          }),
      ...ExperimentPageSideNavGenAIConfig,
      observability: [
        ...ExperimentPageSideNavGenAIConfig.observability,
        {
          label: (
            <FormattedMessage
              defaultMessage="Sessions"
              description="Label for the chat sessions tab in the MLflow experiment navbar"
            />
          ),
          icon: <SpeechBubbleIcon />,
          tabName: ExperimentPageTabName.ChatSessions,
          componentId: 'mlflow.experiment-side-nav.genai.chat-sessions',
        },
      ],
      evaluation: enableScorersUI()
        ? [
            ...ExperimentPageSideNavGenAIConfig.evaluation,
            {
              label: (
                <FormattedMessage
                  defaultMessage="Judges"
                  description="Label for the judges tab in the MLflow experiment navbar"
                />
              ),
              icon: <GavelIcon />,
              tabName: ExperimentPageTabName.Judges,
              componentId: 'mlflow.experiment-side-nav.genai.judges',
            },
          ]
        : ExperimentPageSideNavGenAIConfig.evaluation,
    };
  }

  return ExperimentPageSideNavCustomModelConfig;
};
