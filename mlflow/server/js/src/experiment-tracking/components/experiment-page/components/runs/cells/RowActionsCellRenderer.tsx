import {
  PinIcon,
  PinFillIcon,
  Tooltip,
  VisibleIcon as VisibleHollowIcon,
  VisibleOffIcon,
  useDesignSystemTheme,
  Icon,
  visuallyHidden,
} from '@databricks/design-system';

// TODO: Import this icon from design system when added
import { ReactComponent as VisibleFillIcon } from '../../../../../../common/static/icon-visible-fill.svg';
import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage, defineMessages } from 'react-intl';
import { RunRowType } from '../../../utils/experimentPage.row-types';
import {
  shouldEnableShareExperimentViewByTags,
  shouldUseNewRunRowsVisibilityModel,
} from '../../../../../../common/utils/FeatureUtils';
import { useUpdateExperimentViewUIState } from '../../../contexts/ExperimentPageUIStateContext';
import { RUNS_VISIBILITY_MODE } from '../../../models/ExperimentPageUIStateV2';
import { isRemainingRunsGroup } from '../../../utils/experimentPage.group-row-utils';

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

const VisibleIcon = () =>
  shouldUseNewRunRowsVisibilityModel() ? <Icon component={VisibleFillIcon} /> : <VisibleHollowIcon />;

// Mouse enter/leave delays passed to tooltips are set to 0 so swift toggling/pinning runs is not hampered
const MOUSE_DELAYS = { mouseEnterDelay: 0, mouseLeaveDelay: 0 };

export const RowActionsCellRenderer = React.memo(
  (props: {
    data: RunRowType;
    value: { pinned: boolean; hidden: boolean };
    onTogglePin: (runUuid: string) => void;
    onToggleVisibility: (runUuidOrToggle: string | RUNS_VISIBILITY_MODE, runUuid?: string) => void;
  }) => {
    const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
    const updateUIState = useUpdateExperimentViewUIState();
    const { theme } = useDesignSystemTheme();

    const { groupParentInfo } = props.data;
    const isGroupRow = Boolean(groupParentInfo);
    const { pinned, hidden } = props.value;
    const { runUuid, rowUuid } = props.data;

    // If a row is a run group, we use its rowUuid for setting visibility.
    // If this is a run, use runUuid.
    const runUuidToToggle = groupParentInfo ? rowUuid : runUuid;

    const visibilityMessageDescriptor = isGroupRow
      ? hidden
        ? labels.visibility.groups.unhide
        : labels.visibility.groups.hide
      : hidden
      ? labels.visibility.runs.unhide
      : labels.visibility.runs.hide;

    const pinningMessageDescriptor = isGroupRow
      ? pinned
        ? labels.pinning.groups.unpin
        : labels.pinning.groups.pin
      : pinned
      ? labels.pinning.runs.unpin
      : labels.pinning.runs.pin;

    const displayVisibilityCheckbox =
      !shouldUseNewRunRowsVisibilityModel() ||
      (groupParentInfo && !isRemainingRunsGroup(groupParentInfo)) ||
      (runUuid && !props.data.runDateAndNestInfo?.belongsToGroup);

    return (
      <div css={styles.actionsContainer}>
        {/* Hide/show icon is part of compare runs UI */}
        {!displayVisibilityCheckbox ? (
          <div css={{ width: theme.general.iconFontSize }} />
        ) : (
          <Tooltip
            dangerouslySetAntdProps={MOUSE_DELAYS}
            placement="right"
            title={<FormattedMessage {...visibilityMessageDescriptor} />}
          >
            <label
              css={[
                styles.actionCheckbox(theme),
                // We show this button only in the runs compare mode and only when the feature flag is set
                shouldUseNewRunRowsVisibilityModel() && styles.showOnlyInCompareMode,
              ]}
              className="is-visibility-toggle"
            >
              <span css={visuallyHidden}>
                <FormattedMessage {...visibilityMessageDescriptor} />
              </span>
              <input
                type="checkbox"
                checked={!hidden}
                onChange={() => {
                  if (runUuidToToggle) {
                    if (shouldUseNewRunRowsVisibilityModel()) {
                      props.onToggleVisibility(RUNS_VISIBILITY_MODE.CUSTOM, runUuidToToggle);
                    } else {
                      props.onToggleVisibility(runUuidToToggle);
                    }
                  }
                }}
              />
              {!hidden ? <VisibleIcon /> : <VisibleOffIcon />}
            </label>
          </Tooltip>
        )}
        {((props.data.pinnable && runUuid) || groupParentInfo) && (
          <Tooltip
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
                  // If using new view state model, update the pinned runs in the UI state.
                  // TODO: Remove this once we migrate to the new view state model
                  if (usingNewViewStateModel) {
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
                  } else if (runUuid) {
                    props.onTogglePin(runUuid);
                  }
                }}
              />
              {pinned ? <PinFillIcon /> : <PinIcon />}
            </label>
          </Tooltip>
        )}
      </div>
    );
  },
  (prevProps, nextProps) =>
    prevProps.value.hidden === nextProps.value.hidden && prevProps.value.pinned === nextProps.value.pinned,
);

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
    // Styling for the show/hide button - it uses different color for active/inactive
    '&.is-visibility-toggle svg': {
      color: theme.colors.grey400,
      '.ag-row:hover &': {
        color: theme.colors.grey500,
      },
    },
    '& input:checked + span svg': {
      color: theme.colors.grey500,
    },
  }),
};
