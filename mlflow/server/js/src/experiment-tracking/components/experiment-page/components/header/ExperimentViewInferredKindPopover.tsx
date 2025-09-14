import { Button, CloseIcon, Popover, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';
import type { ExperimentKind } from '../../../../constants';
import { ExperimentKindDropdownLabels, ExperimentKindShortLabels } from '../../../../utils/ExperimentKindUtils';

export const ExperimentViewInferredKindPopover = ({
  children,
  inferredExperimentKind,
  onConfirm,
  onDismiss,
  isInferredKindEditable = false,
}: {
  children: React.ReactNode;
  inferredExperimentKind: ExperimentKind;
  onConfirm?: () => void;
  onDismiss?: () => void;
  isInferredKindEditable?: boolean;
}) => {
  const intl = useIntl();
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'block', position: 'relative' }}>
      {children}
      <Popover.Root componentId="mlflow.experiment_view.header.experiment_kind_inference_popover" open modal={false}>
        <Popover.Trigger asChild>
          <div css={{ position: 'absolute', left: 0, bottom: 0, right: 0, height: 0 }} />
        </Popover.Trigger>
        <Popover.Content css>
          <Popover.Arrow />
          <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
            <div css={{ flex: 1 }}>
              <Typography.Paragraph css={{ maxWidth: 300 }}>
                <FormattedMessage
                  defaultMessage="We've automatically detected the experiment type to be ''{kindLabel}''. {isEditable, select, true {You can either confirm or change the type.} other {}}"
                  description="Popover message for inferred experiment kind"
                  values={{
                    kindLabel: intl.formatMessage(ExperimentKindShortLabels[inferredExperimentKind]),
                    isEditable: isInferredKindEditable,
                  }}
                />
              </Typography.Paragraph>
              <Button
                componentId="mlflow.experiment_view.header.experiment_kind_inference_popover.confirm"
                onClick={onConfirm}
                type="primary"
                size="small"
              >
                <FormattedMessage
                  defaultMessage="Confirm"
                  description="Button label to confirm the inferred experiment kind"
                />
              </Button>
            </div>
            <Button
              componentId="mlflow.experiment_view.header.experiment_kind_inference_popover.dismiss"
              onClick={onDismiss}
              icon={<CloseIcon />}
              size="small"
            />
          </div>
        </Popover.Content>
      </Popover.Root>
    </div>
  );
};
