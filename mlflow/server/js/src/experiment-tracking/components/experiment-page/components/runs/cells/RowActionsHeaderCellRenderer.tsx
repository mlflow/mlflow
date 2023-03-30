import { DropdownMenu, VisibleIcon, VisibleOffIcon } from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React from 'react';
import { FormattedMessage } from 'react-intl';
import { RUNS_VISIBILITY_MODE } from '../../../utils/experimentPage.common-utils';

/**
 * A component used to render "eye" icon in the table header used to hide/show all runs
 */
export const RowActionsHeaderCellRenderer = React.memo(
  (props: { onToggleVisibility: (runUuidOrToggle: string) => void }) => (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button
          css={styles.actionButton}
          data-testid='experiment-view-runs-visibility-column-header'
        >
          <VisibleIcon />
        </button>
      </DropdownMenu.Trigger>

      <DropdownMenu.Content>
        <DropdownMenu.Item
          onClick={() => props.onToggleVisibility(RUNS_VISIBILITY_MODE.HIDEALL)}
          data-testid='experiment-view-runs-visibility-hide-all'
        >
          <DropdownMenu.IconWrapper>
            <VisibleOffIcon />
          </DropdownMenu.IconWrapper>
          <FormattedMessage
            defaultMessage='Hide all runs'
            description='Menu option for hiding all runs in the experiment view runs compare mode'
          />
        </DropdownMenu.Item>
        <DropdownMenu.Item
          onClick={() => props.onToggleVisibility(RUNS_VISIBILITY_MODE.SHOWALL)}
          data-testid='experiment-view-runs-visibility-show-all'
        >
          <DropdownMenu.IconWrapper>
            <VisibleIcon />
          </DropdownMenu.IconWrapper>
          <FormattedMessage
            defaultMessage='Show all runs'
            description='Menu option for revealing all hidden runs in the experiment view runs compare mode'
          />
        </DropdownMenu.Item>
      </DropdownMenu.Content>

      {/*  */}
    </DropdownMenu.Root>
  ),
);

const styles = {
  actionButton: (theme: Theme) => ({
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    padding: '8px',
    // When visibility icon is next to the ag-grid checkbox, remove the bonus padding
    '.ag-checkbox:not(.ag-hidden) + &': { padding: '0 1px' },
    svg: {
      width: 14,
      height: 14,
      cursor: 'pointer',
      color: theme.colors.grey600,
    },
  }),
};
