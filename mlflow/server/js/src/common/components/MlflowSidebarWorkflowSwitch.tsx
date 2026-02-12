import { Typography, useDesignSystemTheme } from '@databricks/design-system';
import { WorkflowType } from '../contexts/WorkflowTypeContext';
import { FormattedMessage } from '@databricks/i18n';

const Pill = ({
  isActive,
  workflowType,
  setWorkflowType,
}: {
  isActive: boolean;
  workflowType: WorkflowType;
  setWorkflowType: (workflowType: WorkflowType) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const borderColor =
    workflowType === WorkflowType.GENAI ? theme.gradients.aiBorderGradient : theme.colors.actionDefaultBorderDefault;
  const label =
    workflowType === WorkflowType.GENAI ? (
      <FormattedMessage defaultMessage="GenAI" description="Label for GenAI workflow type option" />
    ) : (
      <FormattedMessage defaultMessage="Model training" description="Label for model training workflow type option" />
    );

  if (isActive) {
    return (
      <div
        css={{
          position: 'relative',
          background: borderColor,
          margin: -1,
          borderRadius: theme.borders.borderRadiusFull,
          padding: 1,
          cursor: 'pointer',
        }}
        onClick={() => setWorkflowType(workflowType)}
      >
        <div
          css={{
            background: theme.colors.backgroundPrimary,
            borderRadius: theme.borders.borderRadiusFull,
            padding: theme.spacing.xs,
            paddingRight: theme.spacing.md,
            paddingLeft: theme.spacing.md,
          }}
        >
          <Typography.Text>{label}</Typography.Text>
        </div>
      </div>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        cursor: 'pointer',
        // magic number to prevent the text from jumping around
        // when switching between workflow types
        paddingRight: workflowType === WorkflowType.GENAI ? 0 : 7,
        paddingLeft: workflowType === WorkflowType.GENAI ? 7 : 0,
      }}
      onClick={() => setWorkflowType(workflowType)}
    >
      <Typography.Text color="secondary">{label}</Typography.Text>
    </div>
  );
};

export const MlflowSidebarWorkflowSwitch = ({
  workflowType,
  setWorkflowType,
}: {
  workflowType: WorkflowType;
  setWorkflowType: (workflowType: WorkflowType) => void;
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        position: 'relative',
        alignItems: 'center',
        background: theme.colors.backgroundSecondary,
        borderRadius: theme.borders.borderRadiusFull,
        border: `1px solid ${theme.colors.actionDefaultBorderDefault}`,
      }}
    >
      <Pill
        isActive={workflowType === WorkflowType.GENAI}
        workflowType={WorkflowType.GENAI}
        setWorkflowType={setWorkflowType}
      />
      <Pill
        isActive={workflowType === WorkflowType.MACHINE_LEARNING}
        workflowType={WorkflowType.MACHINE_LEARNING}
        setWorkflowType={setWorkflowType}
      />
    </div>
  );
};
