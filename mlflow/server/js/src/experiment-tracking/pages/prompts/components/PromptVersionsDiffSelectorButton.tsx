import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

/**
 * A custom split button to select versions to compare in the prompt details page.
 */
export const PromptVersionsDiffSelectorButton = ({
  isSelectedFirstToCompare,
  isSelectedSecondToCompare,
  onSelectFirst,
  onSelectSecond,
}: {
  isSelectedFirstToCompare: boolean;
  isSelectedSecondToCompare: boolean;
  onSelectFirst?: () => void;
  onSelectSecond?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div
      css={{ width: theme.general.buttonHeight, display: 'flex', alignItems: 'center', paddingRight: theme.spacing.sm }}
    >
      <div css={{ display: 'flex', height: theme.general.buttonInnerHeight + theme.spacing.xs, gap: 0, flex: 1 }}>
        <Tooltip
          componentId="mlflow.prompts.details.select_baseline.tooltip"
          content={
            <FormattedMessage
              defaultMessage="Select as baseline version"
              description="Label for selecting baseline prompt version in the comparison view"
            />
          }
          delayDuration={0}
          side="left"
        >
          <button
            onClick={onSelectFirst}
            role="radio"
            aria-checked={isSelectedFirstToCompare}
            aria-label={intl.formatMessage({
              defaultMessage: 'Select as baseline version',
              description: 'Label for selecting baseline prompt version in the comparison view',
            })}
            css={{
              flex: 1,
              border: `1px solid ${
                isSelectedFirstToCompare
                  ? theme.colors.actionDefaultBorderFocus
                  : theme.colors.actionDefaultBorderDefault
              }`,
              borderRight: 0,
              marginLeft: 1,
              borderTopLeftRadius: theme.borders.borderRadiusMd,
              borderBottomLeftRadius: theme.borders.borderRadiusMd,
              backgroundColor: isSelectedFirstToCompare
                ? theme.colors.actionDefaultBackgroundPress
                : theme.colors.actionDefaultBackgroundDefault,
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
              },
            }}
          />
        </Tooltip>
        <Tooltip
          componentId="mlflow.prompts.details.select_compared.tooltip"
          content={
            <FormattedMessage
              defaultMessage="Select as compared version"
              description="Label for selecting compared prompt version in the comparison view"
            />
          }
          delayDuration={0}
          side="right"
        >
          <button
            onClick={onSelectSecond}
            role="radio"
            aria-checked={isSelectedSecondToCompare}
            aria-label={intl.formatMessage({
              defaultMessage: 'Select as compared version',
              description: 'Label for selecting compared prompt version in the comparison view',
            })}
            css={{
              flex: 1,
              border: `1px solid ${
                isSelectedSecondToCompare
                  ? theme.colors.actionDefaultBorderFocus
                  : theme.colors.actionDefaultBorderDefault
              }`,
              borderLeft: `1px solid ${
                isSelectedFirstToCompare || isSelectedSecondToCompare
                  ? theme.colors.actionDefaultBorderFocus
                  : theme.colors.actionDefaultBorderDefault
              }`,
              borderTopRightRadius: theme.borders.borderRadiusMd,
              borderBottomRightRadius: theme.borders.borderRadiusMd,
              backgroundColor: isSelectedSecondToCompare
                ? theme.colors.actionDefaultBackgroundPress
                : theme.colors.actionDefaultBackgroundDefault,
              cursor: 'pointer',
              '&:hover': {
                backgroundColor: theme.colors.actionDefaultBackgroundHover,
              },
            }}
          />
        </Tooltip>
      </div>
    </div>
  );
};
