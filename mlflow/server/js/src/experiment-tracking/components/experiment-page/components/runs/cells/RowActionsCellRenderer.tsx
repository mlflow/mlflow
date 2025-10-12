import {
  PinIcon,
  PinFillIcon,
  LegacyTooltip,
  VisibleIcon as VisibleHollowIcon,
  VisibleOffIcon,
  useDesignSystemTheme,
  Icon,
  visuallyHidden,
} from '@databricks/design-system';
import type { SuppressKeyboardEventParams } from '@ag-grid-community/core';

// TODO: Import this icon from design system when added
import { ReactComponent as VisibleFillIcon } from '../../../../../../common/static/icon-visible-fill.svg';
import type { Theme } from '@emotion/react';
import React, { useMemo } from 'react';
import { FormattedMessage, defineMessages } from 'react-intl';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import { RunRowVisibilityControl } from '../../../utils/experimentPage.row-types';
import { shouldEnableToggleIndividualRunsInGroups } from '../../../../../../common/utils/FeatureUtils';
import { useUpdateExperimentViewUIState } from '../../../contexts/ExperimentPageUIStateContext';
import type { RUNS_VISIBILITY_MODE } from '../../../models/ExperimentPageUIState';
import { isRemainingRunsGroup } from '../../../utils/experimentPage.group-row-utils';
import { RunVisibilityControlButton } from './RunVisibilityControlButton';
import { useExperimentViewRunsTableHeaderContext } from '../ExperimentViewRunsTableHeaderContext';

const labels = {
  visibility: {
    groups: defineMessages({
      unhide: {
        defaultMessage: 'Unhide group',
        description: 'A tooltip for the visibility icon button in the runs table next to the hidden run group',
      },
      hide: {
        defaultMessage: 'Hide group',
        description: 'A tooltip for the visibility icon button in the runs table next to the visible run group',
      },
    }),
    runs: defineMessages({
      unhide: {
        defaultMessage: 'Unhide run',
        description: 'A tooltip for the visibility icon button in the runs table next to the hidden run',
      },
      hide: {
        defaultMessage: 'Hide run',
        description: 'A tooltip for the visibility icon button in the runs table next to the visible run',
      },
    }),
  },
  pinning: {
    groups: defineMessages({
      unpin: {
        defaultMessage: 'Unpin group',
        description: 'A tooltip for the pin icon button in the runs table next to the pinned run group',
      },
      pin: {
        defaultMessage: 'Pin group',
        description: 'A tooltip for the pin icon button in the runs table next to the not pinned run group',
      },
    }),
    runs: defineMessages({
      unpin: {
        defaultMessage: 'Unpin run',
        description: 'A tooltip for the pin icon button in the runs table next to the pinned run',
      },
      pin: {
        defaultMessage: 'Pin run',
        description: 'A tooltip for the pin icon button in the runs table next to the not pinned run',
      },
    }),
  },
};

// Mouse enter/leave delays passed to tooltips are set to 0 so swift toggling/pinning runs is not hampered
const MOUSE_DELAYS = { mouseEnterDelay: 0, mouseLeaveDelay: 0 };

export const RowActionsCellRenderer = React.memo(
  (props: {
    data: RunRowType;
    value: { pinned: boolean; hidden: boolean };
    onTogglePin: (runUuid: string) => void;
    onToggleVisibility: (runUuidOrToggle: string | RUNS_VISIBILITY_MODE, runUuid?: string) => void;
  }) => {
    const updateUIState = useUpdateExperimentViewUIState();
    const { theme } = useDesignSystemTheme();
    const { useGroupedValuesInCharts } = useExperimentViewRunsTableHeaderContext();

    const { groupParentInfo, runDateAndNestInfo, visibilityControl } = props.data;
    const { belongsToGroup } = runDateAndNestInfo || {};
    const isGroupRow = Boolean(groupParentInfo);
    const isVisibilityButtonDisabled =
      shouldEnableToggleIndividualRunsInGroups() && visibilityControl === RunRowVisibilityControl.Disabled;
    const { pinned, hidden } = props.value;
    const { runUuid, rowUuid } = props.data;

    // If a row is a run group, we use its rowUuid for setting visibility.
    // If this is a run, use runUuid.
    const runUuidToToggle = groupParentInfo ? rowUuid : runUuid;

    const isRowHidden = (() => {
      // If "Use grouping from the runs table in charts" option is off and we're displaying a group,
      // we should check if all runs in the group are hidden in order to determine visibility toggle.
      if (shouldEnableToggleIndividualRunsInGroups() && useGroupedValuesInCharts === false && groupParentInfo) {
        return Boolean(groupParentInfo.allRunsHidden);
      }

      // Otherwise, we should use the hidden flag from the row itself.
      return hidden;
    })();

    const visibilityMessageDescriptor = isGroupRow
      ? isRowHidden
        ? labels.visibility.groups.unhide
        : labels.visibility.groups.hide
      : isRowHidden
      ? labels.visibility.runs.unhide
      : labels.visibility.runs.hide;

    const pinningMessageDescriptor = isGroupRow
      ? pinned
        ? labels.pinning.groups.unpin
        : labels.pinning.groups.pin
      : pinned
      ? labels.pinning.runs.unpin
      : labels.pinning.runs.pin;

    const isVisibilityButtonHidden = useMemo(() => {
      if (shouldEnableToggleIndividualRunsInGroups()) {
        return visibilityControl === RunRowVisibilityControl.Hidden;
      }
      return !((groupParentInfo && !isRemainingRunsGroup(groupParentInfo)) || (Boolean(runUuid) && !belongsToGroup));
    }, [groupParentInfo, belongsToGroup, runUuid, visibilityControl]);

    return (
      <div css={styles.actionsContainer}>
        <RunVisibilityControlButton
          rowHidden={isRowHidden}
          buttonHidden={isVisibilityButtonHidden}
          disabled={isVisibilityButtonDisabled}
          label={<FormattedMessage {...visibilityMessageDescriptor} />}
          onClick={props.onToggleVisibility}
          runUuid={runUuidToToggle}
          css={[
            styles.actionCheckbox(theme),
            // We show this button only in the runs compare mode
            styles.showOnlyInCompareMode,
          ]}
        />
        {((props.data.pinnable && runUuid) || groupParentInfo) && (
          <LegacyTooltip
            dangerouslySetAntdProps={MOUSE_DELAYS}
            placement="right"
            // We have to force remount of the tooltip with every rerender, otherwise it will jump
            // around when the row order changes.
            key={Math.random()}
            title={<FormattedMessage {...pinningMessageDescriptor} />}
          >
            <label css={styles.actionCheckbox(theme)} className="is-pin-toggle" data-testid="column-pin-toggle">
              <span css={visuallyHidden}>
                <FormattedMessage {...pinningMessageDescriptor} />
              </span>
              <input
                type="checkbox"
                checked={pinned}
                onChange={() => {
                  const uuidToPin = groupParentInfo ? props.data.rowUuid : runUuid;
                  updateUIState((existingState) => {
                    if (uuidToPin) {
                      return {
                        ...existingState,
                        runsPinned: !existingState.runsPinned.includes(uuidToPin)
                          ? [...existingState.runsPinned, uuidToPin]
                          : existingState.runsPinned.filter((r) => r !== uuidToPin),
                      };
                    }
                    return existingState;
                  });
                }}
              />
              {pinned ? <PinFillIcon /> : <PinIcon />}
            </label>
          </LegacyTooltip>
        )}
      </div>
    );
  },
  (prevProps, nextProps) =>
    prevProps.value.hidden === nextProps.value.hidden &&
    prevProps.value.pinned === nextProps.value.pinned &&
    prevProps.data.visibilityControl === nextProps.data.visibilityControl &&
    prevProps.data.groupParentInfo?.allRunsHidden === nextProps.data.groupParentInfo?.allRunsHidden,
);

/**
 * A utility function that enables custom keyboard navigation for the row actions cell renderer by providing
 * conditional suppression of default events.
 */
export const RowActionsCellRendererSuppressKeyboardEvents = ({ event }: SuppressKeyboardEventParams) => {
  if (
    event.key === 'Tab' &&
    event.target instanceof HTMLElement &&
    // Let's suppress the default action if the focus is on cell or on visibility toggle checkbox, allowing
    // tab to move to the next focusable element.
    (event.target.classList.contains('ag-cell') || event.target.classList.contains('is-visibility-toggle-checkbox'))
  ) {
    return true;
  }
  return false;
};

const styles = {
  actionsContainer: {
    display: 'flex',
    gap: 18, // In design there's 20 px of gutter, it's minus 2 px due to pin icon's internal padding
  },
  showOnlyInCompareMode: {
    display: 'none',
    '.is-table-comparing-runs-mode &': {
      display: 'flex',
    },
  },
  actionCheckbox: (theme: Theme) => ({
    input: { width: 0, appearance: 'none' as const },
    cursor: 'pointer',
    display: 'flex',
    svg: {
      width: theme.general.iconFontSize,
      height: theme.general.iconFontSize,
      cursor: 'pointer',
    },
    // Styling for the pin button - it's transparent when unpinned and not hovered
    '&.is-pin-toggle svg': {
      color: 'transparent',
      '.ag-row:hover &': {
        color: theme.colors.grey500,
      },
    },
    '& input:checked + span svg': {
      color: theme.colors.grey500,
    },
    '& input:focus-visible + span svg': {
      color: theme.colors.grey500,
    },
  }),
};
