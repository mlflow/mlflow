import { PinBorderIcon, PinFillIcon, Tooltip } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { RunRowType } from '../../../utils/experimentPage.row-types';

export const PinRowCellRenderer = React.memo(
  (props: { data: RunRowType; value: boolean; onTogglePin: any }) => {
    if (!props.data.pinnable) {
      return null;
    }
    return (
      <Tooltip
        placement='right'
        title={
          props.value ? (
            <FormattedMessage
              defaultMessage='Unpin run'
              description='A tooltip for the pin icon button in the runs table next to the pinned run'
            />
          ) : (
            <FormattedMessage
              defaultMessage='Click to pin the run'
              description='A tooltip for the pin icon button in the runs table next to the not pinned run'
            />
          )
        }
      >
        <label css={styles.pinWrapper}>
          <input
            type='checkbox'
            checked={props.value}
            onChange={() => {
              props.onTogglePin(props.data.runUuid);
            }}
          />
          {props.value ? <PinFillIcon /> : <PinBorderIcon />}
        </label>
      </Tooltip>
    );
  },
);

const styles = {
  pinWrapper: (theme: Theme) => ({
    input: { width: 0, appearance: 'none' as const },
    cursor: 'pointer',
    display: 'inline-block',
    svg: {
      width: 14,
      height: 14,
      cursor: 'pointer',
      color: 'transparent',
    },
    // Two selectors below need to be separate - emotion doesn't
    // allow nested "&" in selectors joined by a comma
    '.ag-row-hover & svg': {
      color: theme.colors.grey600,
    },
    'input:checked + span svg': {
      color: theme.colors.grey600,
    },
  }),
};
