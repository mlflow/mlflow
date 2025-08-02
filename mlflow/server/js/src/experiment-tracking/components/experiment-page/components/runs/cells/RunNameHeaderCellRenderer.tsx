// React and third-party
import React, { useMemo, useCallback } from 'react';
import { FormattedMessage, useIntl, IntlShape } from 'react-intl';

// Design system
import {
  SortAscendingIcon,
  SortDescendingIcon,
  useDesignSystemTheme,
  DropdownMenu,
  Button,
  VisibleOffIcon,
  Icon,
  Tooltip,
} from '@databricks/design-system';

// Internal imports
import {
  useUpdateExperimentPageSearchFacets,
  useExperimentPageSearchFacets,
} from '../../../hooks/useExperimentPageSearchFacets';
import { RUN_LIMIT_OPTIONS } from '../../../../../constants';
import { RunNameHeaderCellRendererErrorBoundary } from './RunNameHeaderCellRendererErrorBoundary';
// TODO: Import this icon from design system when added
import { ReactComponent as VisibleFillIcon } from '../../../../../../common/static/icon-visible-fill.svg';

const VisibleIcon = () => <Icon component={VisibleFillIcon} />;

/**
 * Get the appropriate tooltip content based on the visibility state
 */
const getVisibilityTooltipContent = (hideFinishedRuns: boolean, intl: IntlShape) => {
  return hideFinishedRuns
    ? intl.formatMessage({
        defaultMessage: 'Some runs are hidden. Click to show options.',
        description: 'Tooltip for the visibility toggle button when runs are hidden',
      })
    : intl.formatMessage({
        defaultMessage: 'All runs are visible. Click to show options.',
        description: 'Tooltip for the visibility toggle button when all runs are visible',
      });
};

interface ColumnHeaderParams {
  canonicalSortKey?: string;
}

interface ColumnDefinition {
  headerName?: string;
  headerComponentParams?: ColumnHeaderParams;
}

interface AgGridColumn {
  getColDef(): ColumnDefinition;
}

export interface RunNameHeaderCellRendererProps {
  displayName?: string;
  column?: AgGridColumn;
  enableSorting?: boolean;
  context?: {
    orderByKey: string;
    orderByAsc: boolean;
  };
}

const RunNameHeaderCellRendererComponent = (props: RunNameHeaderCellRendererProps) => {
  const { displayName, column, enableSorting = true, context: tableContext } = props;
  const { orderByKey = '', orderByAsc = false } = tableContext || {};
  const updateSearchFacets = useUpdateExperimentPageSearchFacets();
  const [searchFacetsState] = useExperimentPageSearchFacets();
  const { hideFinishedRuns, runLimit } = searchFacetsState || {};

  // Get canonical sort key from column definition
  const canonicalSortKey = column?.getColDef()?.headerComponentParams?.canonicalSortKey;
  const actualDisplayName = displayName || column?.getColDef()?.headerName || 'Run Name';
  const selectedCanonicalSortKey = canonicalSortKey;
  const intl = useIntl();

  // Create stable callbacks to prevent unnecessary re-renders
  const handleRunLimitChange = useCallback(
    (value: string) => {
      let limit: number | null = null;

      if (value === 'all') {
        limit = null;
      } else {
        const parsedValue = parseInt(value, 10);
        // Validate the parsed value is a positive integer
        if (!isNaN(parsedValue) && parsedValue > 0) {
          limit = parsedValue;
        } else {
          // Log error but don't crash - fallback to showing all runs
          console.warn(`Invalid run limit value: ${value}. Falling back to show all runs.`);
          limit = null;
        }
      }

      updateSearchFacets({ runLimit: limit });
    },
    [updateSearchFacets],
  );

  const handleHideFinishedRunsChange = useCallback(
    (checked: boolean) => {
      updateSearchFacets({ hideFinishedRuns: checked });
    },
    [updateSearchFacets],
  );

  // Memoize the dropdown with stable callbacks
  const visibilityDropdown = useMemo(
    () => (
      <DropdownMenu.Root modal={false}>
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            height: '100%',
          }}
        >
          <Tooltip
            componentId="run_name_header_visibility_tooltip"
            content={getVisibilityTooltipContent(hideFinishedRuns || false, intl)}
          >
            <DropdownMenu.Trigger asChild>
              <Button
                componentId="run_name_header_visibility_dropdown"
                icon={hideFinishedRuns ? <VisibleOffIcon /> : <VisibleIcon />}
                aria-label={intl.formatMessage({
                  defaultMessage: 'Toggle visibility of runs',
                  description: 'Experiment page > runs table > toggle visibility of runs > accessible label',
                })}
                type="tertiary"
                size="small"
                css={{
                  height: '24px', // Fixed height to prevent header row expansion
                  minHeight: '24px',
                  flexShrink: 0,
                }}
              />
            </DropdownMenu.Trigger>
          </Tooltip>
        </div>

        <DropdownMenu.Content>
          <DropdownMenu.RadioGroup
            componentId="run_name_header_run_limit_group"
            value={runLimit?.toString() || 'all'}
            onValueChange={handleRunLimitChange}
          >
            <DropdownMenu.RadioItem value={RUN_LIMIT_OPTIONS.FIRST_10.toString()}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show first 10"
                description="Menu option for showing only the first 10 runs"
              />
            </DropdownMenu.RadioItem>
            <DropdownMenu.RadioItem value={RUN_LIMIT_OPTIONS.FIRST_20.toString()}>
              <DropdownMenu.ItemIndicator />
              <FormattedMessage
                defaultMessage="Show first 20"
                description="Menu option for showing only the first 20 runs"
              />
            </DropdownMenu.RadioItem>
            <DropdownMenu.RadioItem value="all">
              <DropdownMenu.ItemIndicator />
              <FormattedMessage defaultMessage="Show all runs" description="Menu option for showing all runs" />
            </DropdownMenu.RadioItem>
          </DropdownMenu.RadioGroup>

          <DropdownMenu.Separator />

          <DropdownMenu.CheckboxItem
            componentId="run_name_header_hide_finished_checkbox"
            checked={hideFinishedRuns}
            onCheckedChange={handleHideFinishedRunsChange}
            aria-label={
              hideFinishedRuns
                ? intl.formatMessage({
                    defaultMessage: 'Disable hiding of finished runs with FINISHED, FAILED, and KILLED statuses',
                    description: 'Accessible label for hide finished runs checkbox when checked',
                  })
                : intl.formatMessage({
                    defaultMessage: 'Hide finished runs with FINISHED, FAILED, and KILLED statuses',
                    description: 'Accessible label for hide finished runs checkbox when unchecked',
                  })
            }
          >
            <DropdownMenu.ItemIndicator />
            <FormattedMessage
              defaultMessage="Hide finished runs"
              description="Menu option for hiding finished runs (FINISHED, FAILED, KILLED status) - text stays constant, checkmark indicates state"
            />
          </DropdownMenu.CheckboxItem>
        </DropdownMenu.Content>
      </DropdownMenu.Root>
    ),
    [hideFinishedRuns, runLimit, intl, handleRunLimitChange, handleHideFinishedRunsChange],
  );

  const handleSortBy = () => {
    let newOrderByAsc = !orderByAsc;

    // If the new sortKey is not equal to the previous sortKey, reset the orderByAsc
    if (selectedCanonicalSortKey !== orderByKey) {
      newOrderByAsc = false;
    }
    updateSearchFacets({ orderByKey: selectedCanonicalSortKey, orderByAsc: newOrderByAsc });
  };

  const { theme } = useDesignSystemTheme();
  const isOrderedByClassName = 'is-ordered-by';

  return (
    <div
      role="columnheader"
      css={{
        height: '100%',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <div
        css={{
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          overflow: 'hidden',
          paddingLeft: theme.spacing.xs + theme.spacing.sm,
          paddingRight: theme.spacing.xs + theme.spacing.sm,
          gap: theme.spacing.xs,
          flex: 1,
          svg: {
            color: theme.colors.textSecondary,
          },
          '&:hover': {
            color: enableSorting ? theme.colors.actionTertiaryTextHover : 'unset',
            svg: {
              color: theme.colors.actionTertiaryTextHover,
            },
          },
        }}
        className={selectedCanonicalSortKey === orderByKey ? isOrderedByClassName : ''}
      >
        <div
          css={{
            display: 'flex',
            alignItems: 'center',
            gap: theme.spacing.xs,
            cursor: enableSorting ? 'pointer' : 'default',
          }}
          onClick={enableSorting ? handleSortBy : undefined}
        >
          <span data-testid={`sort-header-${actualDisplayName}`}>{actualDisplayName}</span>
          {enableSorting && selectedCanonicalSortKey === orderByKey ? (
            orderByAsc ? (
              <SortAscendingIcon />
            ) : (
              <SortDescendingIcon />
            )
          ) : null}
        </div>

        {/* Runs Visibility Dropdown - positioned right next to the run name */}
        {visibilityDropdown}
      </div>
    </div>
  );
};

// Simple memoization for performance with error boundary
const MemoizedRunNameHeaderCellRenderer = React.memo(RunNameHeaderCellRendererComponent);

export const RunNameHeaderCellRenderer: React.FC<RunNameHeaderCellRendererProps> = (props) => (
  <RunNameHeaderCellRendererErrorBoundary>
    <MemoizedRunNameHeaderCellRenderer {...props} />
  </RunNameHeaderCellRendererErrorBoundary>
);
