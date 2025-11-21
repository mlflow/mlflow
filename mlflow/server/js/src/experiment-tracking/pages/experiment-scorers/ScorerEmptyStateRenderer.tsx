import React from 'react';
import {
  useDesignSystemTheme,
  Empty,
  Button,
  PlusIcon,
  Spacer,
  SparkleIcon,
  Typography,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { COMPONENT_ID_PREFIX } from './constants';

const getProductionMonitoringDocUrl = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/';
};

interface ScorerEmptyStateRendererProps {
  onAddScorerClick: () => void;
}

const ScorerEmptyStateRenderer: React.FC<ScorerEmptyStateRendererProps> = ({ onAddScorerClick }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
        padding: theme.spacing.lg,
      }}
    >
      <Empty
        image={<SparkleIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />}
        title={
          <FormattedMessage
            defaultMessage="Add a scorer to your experiment to measure your GenAI app quality"
            description="Title for the empty state when no scorers exist"
          />
        }
        description={
          <div css={{ maxWidth: 600, textAlign: 'center' }}>
            <Spacer size="sm" />
            <FormattedMessage
              defaultMessage="Choose from a selection of 8 built-in LLM scorers by Databricks or create your own custom code based scorer. {learnMore}"
              description="Description for the empty state when no scorers exist"
              values={{
                learnMore: (
                  <Typography.Link
                    componentId={`${COMPONENT_ID_PREFIX}.empty-state-learn-more-link`}
                    href={getProductionMonitoringDocUrl()}
                    openInNewTab
                  >
                    <FormattedMessage
                      defaultMessage="Learn more"
                      description="Link text for production monitoring documentation"
                    />
                  </Typography.Link>
                ),
              }}
            />
          </div>
        }
        button={
          <Button
            icon={<PlusIcon />}
            componentId={`${COMPONENT_ID_PREFIX}.empty-state-add-scorer-button`}
            onClick={onAddScorerClick}
          >
            <FormattedMessage defaultMessage="New scorer" description="Button text to add a scorer from empty state" />
          </Button>
        }
      />
    </div>
  );
};

export default ScorerEmptyStateRenderer;
