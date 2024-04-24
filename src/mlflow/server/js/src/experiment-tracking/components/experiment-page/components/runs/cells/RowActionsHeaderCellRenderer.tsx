import {
  DropdownMenu,
  Icon,
  VisibleIcon as VisibleHollowIcon,
  VisibleOffIcon,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { Theme } from '@emotion/react';
import React, { useEffect } from 'react';
import { FormattedMessage, useIntl } from 'react-intl';
import { RUNS_VISIBILITY_MODE } from 'experiment-tracking/components/experiment-page/models/ExperimentPageUIStateV2';
import { shouldUseNewRunRowsVisibilityModel } from '../../../../../../common/utils/FeatureUtils';
// TODO: Import this icon from design system when added
import { ReactComponent as VisibleFillIcon } from '../../../../../../common/static/icon-visible-fill.svg';
import { useExperimentViewRunsTableHeaderContext } from '../ExperimentViewRunsTableHeaderContext';

const VisibleIcon = () =>
  shouldUseNewRunRowsVisibilityModel() ? <Icon component={VisibleFillIcon} /> : <VisibleHollowIcon />;

export const RowActionsHeaderCellRendererV2 = React.memo(
  ({
    allRunsHidden,
    onToggleVisibility,
  }: {
    allRunsHidden?: boolean;
    onToggleVisibility: (mode: RUNS_VISIBILITY_MODE | string, runOrGroupUuid?: string) => void;
  }) => {
    const { theme } = useDesignSystemTheme();
    const intl = useIntl();
    const { runsHiddenMode } = useExperimentViewRunsTableHeaderContext();

    return (
      <DropdownMenu.Root modal={false}>
        <DropdownMenu.Trigger asChild>
          <button
            css={[
              styles.actionButton(theme),
              // We show this button only in the runs compare mode and only when the feature flag is set
              shouldUseNewRunRowsVisibilityModel() && styles.showOnlyInCompareMode,
            ]}
            data-testid="experiment-view-runs-visibility-column-header"
            aria-label={intl.formatMessage({
              defaultMessage: 'Toggle visibility of runs',
              description: 'Experiment page > runs table > toggle visibility of runs > accessible label',
            })}
          >
            {runsHiddenMode === RUNS_VISIBILITY_MODE.HIDEALL || allRunsHidden ? <VisibleOffIcon /> : <VisibleIcon />}
          </button>
        </DropdownMenu.Trigger>

        <DropdownMenu.Content>
          <DropdownMenu.RadioGroup value={runsHiddenMode} onValueChange={(e) => onToggleVisibility(e)}>
            <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.FIRST_10_RUNS}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show first 10"
                description="Menu option for showing only 10 first runs in the experiment view runs compare mode"
              />
            </DropdownMenu.RadioItem>
            <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.FIRST_20_RUNS}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show first 20"
                description="Menu option for showing only 10 first runs in the experiment view runs compare mode"
              />
            </DropdownMenu.RadioItem>
            <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.SHOWALL}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show all runs"
                description="Menu option for revealing all hidden runs in the experiment view runs compare mode"
              />
            </DropdownMenu.RadioItem>
            <DropdownMenu.RadioItem value={RUNS_VISIBILITY_MODE.HIDEALL}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Hide all runs"
                description="Menu option for revealing all hidden runs in the experiment view runs compare mode"
              />
            </DropdownMenu.RadioItem>
          </DropdownMenu.RadioGroup>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    );
  },
);

/**
 * A component used to render "eye" icon in the table header used to hide/show all runs
 */
export const RowActionsHeaderCellRenderer = React.memo(
  (props: {
    allRunsHidden?: boolean;
    onToggleVisibility: (runUuidOrToggle: string) => void;
    eGridHeader?: HTMLElement;
  }) => {
    const intl = useIntl();

    // Since ag-grid does not add accessible labels to its checkboxes, we do it manually.
    // This is executed once per table lifetime.
    useEffect(() => {
      // Find a checkbox in the header
      const selectAllCheckbox = props.eGridHeader?.querySelector('input');

      // If found, assign aria-label attribute
      if (selectAllCheckbox) {
        selectAllCheckbox.ariaLabel = intl.formatMessage({
          defaultMessage: 'Select all runs',
          description: 'Experiment page > runs table > select all rows > accessible label',
        });
      }
    }, [props.eGridHeader, intl]);

    return shouldUseNewRunRowsVisibilityModel() ? (
      <RowActionsHeaderCellRendererV2 {...props} />
    ) : (
      <DropdownMenu.Root modal={false}>
        <DropdownMenu.Trigger asChild>
          <button css={[styles.actionButton]} data-testid="experiment-view-runs-visibility-column-header">
            {props.allRunsHidden ? <VisibleOffIcon /> : <VisibleIcon />}
          </button>
        </DropdownMenu.Trigger>

        <DropdownMenu.Content>
          <DropdownMenu.Item
            onClick={() => props.onToggleVisibility(RUNS_VISIBILITY_MODE.HIDEALL)}
            data-testid="experiment-view-runs-visibility-hide-all"
          >
            <DropdownMenu.IconWrapper>
              <VisibleOffIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Hide all runs"
              description="Menu option for hiding all runs in the experiment view runs compare mode"
            />
          </DropdownMenu.Item>
          <DropdownMenu.Item
            onClick={() => props.onToggleVisibility(RUNS_VISIBILITY_MODE.SHOWALL)}
            data-testid="experiment-view-runs-visibility-show-all"
          >
            <DropdownMenu.IconWrapper>
              <VisibleIcon />
            </DropdownMenu.IconWrapper>
            <FormattedMessage
              defaultMessage="Show all runs"
              description="Menu option for revealing all hidden runs in the experiment view runs compare mode"
            />
          </DropdownMenu.Item>
        </DropdownMenu.Content>

        {/*  */}
      </DropdownMenu.Root>
    );
  },
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
      width: theme.general.iconFontSize,
      height: theme.general.iconFontSize,
      cursor: 'pointer',
      color: theme.colors.grey500,
    },
  }),
  showOnlyInCompareMode: {
    display: 'none',
    '.is-table-comparing-runs-mode &': {
      display: 'flex',
    },
  },
};
