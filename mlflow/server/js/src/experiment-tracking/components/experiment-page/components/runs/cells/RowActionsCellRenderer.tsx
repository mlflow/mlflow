import {
  PinIcon,
  PinFillIcon,
  Tooltip,
  VisibleIcon,
  VisibleOffIcon,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { shouldUseNextRunsComparisonUI } from '../../../../../../common/utils/FeatureUtils';
import { RunRowType } from '../../../utils/experimentPage.row-types';

// Mouse enter/leave delays passed to tooltips are set to 0 so swift toggling/pinning runs is not hampered
const MOUSE_DELAYS = { mouseEnterDelay: 0, mouseLeaveDelay: 0 };

export const RowActionsCellRenderer = React.memo(
  (props: {
    data: RunRowType;
    value: { pinned: boolean; hidden: boolean };
    onTogglePin: (runUuid: string) => void;
    onToggleVisibility: (runUuid: string) => void;
  }) => {
    const { pinned, hidden } = props.value;
    return (
      <div css={styles.actionsContainer}>
        {/* Hide/show icon is part of compare runs UI */}
        {shouldUseNextRunsComparisonUI() && (
          <Tooltip
            dangerouslySetAntdProps={MOUSE_DELAYS}
            placement='right'
            title={
              hidden ? (
                <FormattedMessage
                  defaultMessage='Unhide run'
                  description='A tooltip for the visibility icon button in the runs table next to the hidden run'
                />
              ) : (
                <FormattedMessage
                  defaultMessage='Hide run'
                  description='A tooltip for the visibility icon button in the runs table next to the visible run'
                />
              )
            }
          >
            <label css={styles.actionCheckbox} className='is-visibility-toggle'>
              <input
                type='checkbox'
                checked={!hidden}
                onChange={() => {
                  props.onToggleVisibility(props.data.runUuid);
                }}
              />
              {!hidden ? <VisibleIcon /> : <VisibleOffIcon />}
            </label>
          </Tooltip>
        )}
        {props.data.pinnable && (
          <Tooltip
            dangerouslySetAntdProps={MOUSE_DELAYS}
            placement='right'
            title={
              pinned ? (
                <FormattedMessage
                  defaultMessage='Unpin run'
                  description='A tooltip for the pin icon button in the runs table next to the pinned run'
                />
              ) : (
                <FormattedMessage
                  defaultMessage='Pin run'
                  description='A tooltip for the pin icon button in the runs table next to the not pinned run'
                />
              )
            }
          >
            <label
              css={styles.actionCheckbox}
              className='is-pin-toggle'
              data-testid='column-pin-toggle'
            >
              <input
                type='checkbox'
                checked={pinned}
                onChange={() => {
                  props.onTogglePin(props.data.runUuid);
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
    prevProps.value.hidden === nextProps.value.hidden &&
    prevProps.value.pinned === nextProps.value.pinned,
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
