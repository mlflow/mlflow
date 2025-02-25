import {
  ArrowLeftIcon,
  Button,
  Modal,
  Spacer,
  Input,
  Typography,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { PROMPT_TEMPLATE_EXAMPLES } from '@mlflow/mlflow/src/experiment-tracking/components/evaluation-artifacts-compare/utils/PromptExamples';

const { TextArea } = Input;
type Props = {
  isOpen: boolean;
  closeExamples: () => void;
  closeModal: () => void;
  updatePromptTemplate: (prompt: string) => void;
  updateInputVariableValue: (name: string, value: string) => void;
};

export const EvaluationCreatePromptRunModalExamples = ({
  isOpen,
  closeExamples,
  closeModal,
  updatePromptTemplate,
  updateInputVariableValue,
}: Props): JSX.Element => {
  const { theme } = useDesignSystemTheme();

  const tryPromptTemplate = (promptTemplate: { prompt: string[]; variables: { name: string; value: string }[] }) => {
    updatePromptTemplate(promptTemplate.prompt.join('\n'));
    promptTemplate.variables.forEach(({ name, value }) => {
      updateInputVariableValue(name, value);
    });
    closeExamples();
  };

  return (
    <Modal
      componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodalexamples.tsx_42"
      verticalSizing="maxed_out"
      visible={isOpen}
      onCancel={closeModal}
      title={
        <div>
          <Typography.Title level={2} css={{ marginTop: theme.spacing.sm, marginBottom: theme.spacing.xs }}>
            <Button
              componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodalexamples.tsx_48"
              css={{ marginRight: theme.spacing.sm, marginBottom: theme.spacing.sm }}
              icon={<ArrowLeftIcon />}
              onClick={closeExamples}
            />
            <FormattedMessage
              defaultMessage="Prompt template examples"
              description="Experiment page > new run modal > prompt examples > modal title"
            />
          </Typography.Title>
        </div>
      }
      dangerouslySetAntdProps={{ width: 1200 }}
    >
      {PROMPT_TEMPLATE_EXAMPLES.map((promptTemplate) => (
        <div
          key={promptTemplate.prompt.join('\n')}
          css={{
            display: 'flex',
            flexDirection: 'column',
            gap: theme.spacing.md,
          }}
        >
          <div
            css={{
              boxSizing: 'border-box',
              border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
              borderRadius: theme.legacyBorders.borderRadiusMd,
              background: theme.colors.backgroundPrimary,
              padding: theme.spacing.md,
              margin: 0,
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
              gap: `${theme.spacing.xs}px`,
            }}
          >
            <Typography.Title level={4}>
              <FormattedMessage
                defaultMessage="Prompt Template"
                description="Experiment page > new run modal > prompt examples > prompt template title"
              />
              <Button
                componentId="codegen_mlflow_app_src_experiment-tracking_components_evaluation-artifacts-compare_evaluationcreatepromptrunmodalexamples.tsx_90"
                type="tertiary"
                size="small"
                style={{ float: 'right' }}
                onClick={() => tryPromptTemplate(promptTemplate)}
              >
                <FormattedMessage
                  defaultMessage="Try this template"
                  description="Experiment page > new run modal > prompt examples > try template button"
                />
              </Button>
            </Typography.Title>
            {promptTemplate.prompt.map((line) => (
              <Typography.Paragraph key={line}>{line}</Typography.Paragraph>
            ))}
            <div
              css={{
                marginTop: theme.spacing.xs,
                marginBottom: theme.spacing.xs,
                borderTop: `1px solid ${theme.colors.border}`,
                opacity: 0.5,
              }}
            />
            {promptTemplate.variables.map(({ name, value }) => (
              <div key={name}>
                <Typography.Title level={4}>{name}</Typography.Title>
                <Typography.Paragraph>{value}</Typography.Paragraph>
              </div>
            ))}
          </div>
          <Spacer />
        </div>
      ))}
    </Modal>
  );
};
