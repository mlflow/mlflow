import { Icon, LegacyTooltip, VisibleOffIcon, useDesignSystemTheme, visuallyHidden } from '@databricks/design-system';
import { RUNS_VISIBILITY_MODE } from '../../../models/ExperimentPageUIState';
import { ReactComponent as VisibleFillIcon } from '../../../../../../common/static/icon-visible-fill.svg';
import type { Theme } from '@emotion/react';

const VisibleIcon = () => <Icon component={VisibleFillIcon} />;

interface RunVisibilityControlButtonProps {
  className?: string;
  runUuid: string;
  rowHidden: boolean;
  buttonHidden: boolean;
  disabled: boolean;
  onClick: (runUuidOrToggle: string | RUNS_VISIBILITY_MODE, runUuid?: string, isRowVisible?: boolean) => void;
  label: React.ReactNode;
}

// Mouse enter/leave delays passed to tooltips are set to 0 so swift toggling/pinning runs is not hampered
const MOUSE_DELAYS = { mouseEnterDelay: 0, mouseLeaveDelay: 0 };

export const RunVisibilityControlButton = ({
  runUuid,
  className,
  rowHidden,
  buttonHidden,
  disabled,
  onClick,
  label,
}: RunVisibilityControlButtonProps) => {
  const { theme } = useDesignSystemTheme();
  if (buttonHidden) {
    return <div className={className} css={[styles.button(theme)]} />;
  }
  if (disabled) {
    return (
      <VisibleOffIcon
        className={className}
        css={[
          styles.button(theme),
          {
            opacity: 0.25,
            color: theme.colors.grey400,
          },
        ]}
      />
    );
  }
  return (
    <LegacyTooltip dangerouslySetAntdProps={MOUSE_DELAYS} placement="right" title={label}>
      <label className={className} css={styles.button(theme)}>
        <span css={visuallyHidden}>{label}</span>
        <input
          type="checkbox"
          className="is-visibility-toggle-checkbox"
          checked={!rowHidden}
          onChange={() => {
            if (runUuid) {
              const isRowVisible = !rowHidden;
              onClick(RUNS_VISIBILITY_MODE.CUSTOM, runUuid, isRowVisible);
            }
          }}
        />
        {!rowHidden ? <VisibleIcon /> : <VisibleOffIcon />}
      </label>
    </LegacyTooltip>
  );
};

const styles = {
  button: (theme: Theme) => ({
    width: theme.general.iconFontSize,
    color: theme.colors.grey400,
    '.ag-row:hover &': {
      color: theme.colors.grey500,
    },
  }),
};
