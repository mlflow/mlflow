import { Tooltip, useDesignSystemTheme } from '@databricks/design-system';

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
  return (
    <div
      css={{ width: theme.general.buttonHeight, display: 'flex', alignItems: 'center', paddingRight: theme.spacing.sm }}
    >
      <div css={{ display: 'flex', height: theme.general.buttonInnerHeight + theme.spacing.xs, gap: 0, flex: 1 }}>
        <Tooltip componentId="TODO" content="Select as baseline version" delayDuration={0} side="left">
          <button
            onClick={onSelectFirst}
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
        <Tooltip componentId="TODO" content="Select as compared version" delayDuration={0} side="right">
          <button
            onClick={onSelectSecond}
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
