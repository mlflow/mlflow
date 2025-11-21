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
import { shouldEnableChatSessionsTab } from '@mlflow/mlflow/src/common/utils/FeatureUtils';

export const FULL_WIDTH_CLASS_NAME = 'mlflow-experiment-page-side-nav-full';
export const COLLAPSED_CLASS_NAME = 'mlflow-experiment-page-side-nav-collapsed';

export type ExperimentPageSideNavItem = {
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
    },
  ],
  'prompts-versions': [
    {
      label: (
        <FormattedMessage
          defaultMessage="Agent versions"
          description="Label for the agent versions tab in the MLflow experiment navbar"
        />
      ),
      icon: <ModelsIcon />,
      tabName: ExperimentPageTabName.Models,
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
          defaultMessage="Versions"
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
    const baseConfig = {
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
              },
            ],
          }
        : {
            'top-level': [],
          }),
      ...ExperimentPageSideNavGenAIConfig,
    };

    if (shouldEnableChatSessionsTab()) {
      baseConfig.observability = [
        ...baseConfig.observability,
        {
          label: (
            <FormattedMessage
              defaultMessage="Sessions"
              description="Label for the chat sessions tab in the MLflow experiment navbar"
            />
          ),
          icon: <SpeechBubbleIcon />,
          tabName: ExperimentPageTabName.ChatSessions,
        },
      ];
    }

    return baseConfig;
  }

  return ExperimentPageSideNavCustomModelConfig;
};
