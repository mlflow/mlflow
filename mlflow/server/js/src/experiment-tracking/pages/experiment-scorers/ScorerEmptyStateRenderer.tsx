import React from 'react';
import {
  useDesignSystemTheme,
  Empty,
  Button,
  PlusIcon,
  CodeIcon,
  Spacer,
  GavelIcon,
  Typography,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { COMPONENT_ID_PREFIX } from './constants';

const getProductionMonitoringDocUrl = () => {
  return 'https://mlflow.org/docs/latest/genai/eval-monitor/';
};

interface ScorerEmptyStateRendererProps {
  onAddLLMScorerClick: () => void;
  onAddCustomCodeScorerClick: () => void;
}

const ScorerEmptyStateRenderer: React.FC<ScorerEmptyStateRendererProps> = ({
  onAddLLMScorerClick,
  onAddCustomCodeScorerClick,
}) => {
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
        image={<GavelIcon css={{ fontSize: 48, color: theme.colors.textSecondary }} />}
        title={
          <FormattedMessage
            defaultMessage="Add a judge to your experiment to measure your GenAI app quality"
            description="Title for the empty state when no judges exist"
          />
        }
        description={
          <div css={{ maxWidth: 600, textAlign: 'center' }}>
            <Spacer size="sm" />
            <FormattedMessage
              defaultMessage="Choose from a selection of built-in LLM judges or create your own custom code based judge. {learnMore}"
              description="Description for the empty state when no judges exist"
              values={{
                learnMore: (
                  <Typography.Link
                    componentId="codegen_no_dynamic_mlflow_web_js_src_experiment_tracking_pages_experiment_scorers_scoreremptystaterenderer_59"
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
          <div css={{ display: 'flex', gap: theme.spacing.sm }}>
            <Button
              type="primary"
              icon={<PlusIcon />}
              componentId={`${COMPONENT_ID_PREFIX}.empty-state-add-llm-scorer-button`}
              onClick={onAddLLMScorerClick}
            >
              <FormattedMessage
                defaultMessage="New LLM judge"
                description="Button text to add an LLM judge from empty state"
              />
            </Button>
            <Button
              icon={<CodeIcon />}
              componentId={`${COMPONENT_ID_PREFIX}.empty-state-add-custom-code-scorer-button`}
              onClick={onAddCustomCodeScorerClick}
            >
              <FormattedMessage
                defaultMessage="New custom code judge"
                description="Button text to add a custom code judge from empty state"
              />
            </Button>
          </div>
        }
      />
    </div>
  );
};

export default ScorerEmptyStateRenderer;
