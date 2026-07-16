import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage, useIntl } from 'react-intl';

export const MCPServerVersionDiffSelectorButton = ({
  isSelectedBaseline,
  isSelectedCompared,
  onSelectBaseline,
  onSelectCompared,
}: {
  isSelectedBaseline: boolean;
  isSelectedCompared: boolean;
  onSelectBaseline?: () => void;
  onSelectCompared?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  return (
    <div
      css={{ width: theme.general.buttonHeight, display: 'flex', alignItems: 'center', paddingRight: theme.spacing.sm }}
    >
      <div
        role="radiogroup"
        aria-label={intl.formatMessage({
          defaultMessage: 'Select version for comparison',
          description: 'Aria label for the version comparison radio group',
        })}
        css={{ display: 'flex', height: theme.general.buttonInnerHeight + theme.spacing.xs, gap: 0, flex: 1 }}
      >
        <Tooltip
          componentId="mlflow.mcp_registry.detail.select_baseline.tooltip"
          content={
            <FormattedMessage
              defaultMessage="Select as baseline version"
              description="Label for selecting baseline MCP server version in the comparison view"
            />
          }
          delayDuration={0}
          side="left"
        >
          <button
            onClick={onSelectBaseline}
            role="radio"
            aria-checked={isSelectedBaseline}
            aria-label={intl.formatMessage({
              defaultMessage: 'Select as baseline version',
              description: 'Label for selecting baseline MCP server version in the comparison view',
            })}
            css={{
              flex: 1,
              border: `1px solid ${
                isSelectedBaseline ? theme.colors.actionDefaultBorderFocus : theme.colors.actionDefaultBorderDefault
              }`,
              borderRight: 0,
              marginLeft: 1,
              borderTopLeftRadius: theme.borders.borderRadiusMd,
              borderBottomLeftRadius: theme.borders.borderRadiusMd,
              backgroundColor: isSelectedBaseline
                ? theme.colors.actionDefaultBackgroundPress
                : theme.colors.actionDefaultBackgroundDefault,
              cursor: 'pointer',
              '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
            }}
          />
        </Tooltip>
        <Tooltip
          componentId="mlflow.mcp_registry.detail.select_compared.tooltip"
          content={
            <FormattedMessage
              defaultMessage="Select as compared version"
              description="Label for selecting compared MCP server version in the comparison view"
            />
          }
          delayDuration={0}
          side="right"
        >
          <button
            onClick={onSelectCompared}
            role="radio"
            aria-checked={isSelectedCompared}
            aria-label={intl.formatMessage({
              defaultMessage: 'Select as compared version',
              description: 'Label for selecting compared MCP server version in the comparison view',
            })}
            css={{
              flex: 1,
              border: `1px solid ${
                isSelectedCompared ? theme.colors.actionDefaultBorderFocus : theme.colors.actionDefaultBorderDefault
              }`,
              borderLeft: `1px solid ${
                isSelectedBaseline || isSelectedCompared
                  ? theme.colors.actionDefaultBorderFocus
                  : theme.colors.actionDefaultBorderDefault
              }`,
              borderTopRightRadius: theme.borders.borderRadiusMd,
              borderBottomRightRadius: theme.borders.borderRadiusMd,
              backgroundColor: isSelectedCompared
                ? theme.colors.actionDefaultBackgroundPress
                : theme.colors.actionDefaultBackgroundDefault,
              cursor: 'pointer',
              '&:hover': { backgroundColor: theme.colors.actionDefaultBackgroundHover },
            }}
          />
        </Tooltip>
      </div>
    </div>
  );
};
