import { PinIcon, PinFillIcon, Tooltip, VisibleIcon, VisibleOffIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage, defineMessages } from 'react-intl';
import { RunRowType } from '../../../utils/experimentPage.row-types';
import { shouldEnableShareExperimentViewByTags } from '../../../../../../common/utils/FeatureUtils';
import { useUpdateExperimentViewUIState } from '../../../contexts/ExperimentPageUIStateContext';

const labels = {
  visibility: {
    groups: defineMessages({
      unhide: {
        defaultMessage: 'Unhide runs',
        description: 'A tooltip for the visibility icon button in the runs table next to the hidden run group',
      },
      hide: {
        defaultMessage: 'Hide runs',
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
    onToggleVisibility: (runUuid: string) => void;
  }) => {
    const usingNewViewStateModel = shouldEnableShareExperimentViewByTags();
    const updateUIState = useUpdateExperimentViewUIState();

    const { groupParentInfo } = props.data;
    const isGroupRow = Boolean(groupParentInfo);
    const { pinned, hidden } = props.value;
    const { runUuid } = props.data;

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

    return (
      <div css={styles.actionsContainer}>
        {/* Hide/show icon is part of compare runs UI */}
        {(groupParentInfo || runUuid) && (
          <Tooltip
            dangerouslySetAntdProps={MOUSE_DELAYS}
            placement="right"
            title={<FormattedMessage {...visibilityMessageDescriptor} />}
          >
            <label css={styles.actionCheckbox} className="is-visibility-toggle">
              <input
                type="checkbox"
                checked={!hidden}
                onChange={() => {
                  if (runUuid) {
                    props.onToggleVisibility(runUuid);
                  } else if (groupParentInfo) {
                    updateUIState((existingState) => {
                      if (groupParentInfo.runUuids.every((runUuid) => existingState.runsHidden.includes(runUuid))) {
                        return {
                          ...existingState,
                          runsHidden: existingState.runsHidden.filter(
                            (runUuid) => !groupParentInfo.runUuids.includes(runUuid),
                          ),
                        };
                      }
                      return {
                        ...existingState,
                        runsHidden: [...existingState.runsHidden, ...groupParentInfo.runUuids],
                      };
                    });
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
            <label css={styles.actionCheckbox} className="is-pin-toggle" data-testid="column-pin-toggle">
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
  actionsContainer: () => ({
    display: 'flex',
    gap: 18, // In design there's 20 px of gutter, it's minus 2 px due to pin icon's internal padding
  }),
  actionCheckbox: (theme: Theme) => ({
    input: { width: 0, appearance: 'none' as const },
    cursor: 'pointer',
    display: 'flex',
    svg: {
      width: 14,
      height: 14,
      cursor: 'pointer',
    },
    // Styling for the pin button - it's transparent when unpinned and not hovered
    '&.is-pin-toggle svg': {
      color: 'transparent',
      '.ag-row:hover &': {
        color: theme.colors.grey600,
      },
    },
    // Styling for the show/hide button - it uses different color for active/inactive
    '&.is-visibility-toggle svg': {
      color: theme.colors.grey400,
      '.ag-row:hover &': {
        color: theme.colors.grey600,
      },
    },
    '& input:checked + span svg': {
      color: theme.colors.grey600,
    },
  }),
};
