import type { ICellRendererParams } from '@ag-grid-community/core';
import type { RunRowType } from '../../../utils/experimentPage.row-types';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  NewWindowIcon,
  Tag,
  Tooltip,
  useDesignSystemTheme,
} from '@databricks/design-system';
import {
  createSearchFilterFromRunGroupInfo,
  getRunGroupDisplayName,
  isRemainingRunsGroup,
} from '../../../utils/experimentPage.group-row-utils';
import { useUpdateExperimentViewUIState } from '../../../contexts/ExperimentPageUIStateContext';
import { useCallback, useMemo } from 'react';
import { RunColorPill } from '../../RunColorPill';
import invariant from 'invariant';
import { FormattedMessage } from 'react-intl';
import { useGetExperimentRunColor, useSaveExperimentRunColor } from '../../../hooks/useExperimentRunColor';
import { useExperimentViewRunsTableHeaderContext } from '../ExperimentViewRunsTableHeaderContext';
import { shouldEnableToggleIndividualRunsInGroups } from '../../../../../../common/utils/FeatureUtils';
import type { To } from '../../../../../../common/utils/RoutingUtils';
import { Link, useLocation } from '../../../../../../common/utils/RoutingUtils';
import { EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW } from '../../../hooks/useExperimentPageSearchFacets';

export interface GroupParentCellRendererProps extends ICellRendererParams {
  data: RunRowType;
  isComparingRuns?: boolean;
}

export const GroupParentCellRenderer = ({ data, isComparingRuns }: GroupParentCellRendererProps) => {
  const groupParentInfo = data.groupParentInfo;
  const hidden = data.hidden;
  invariant(groupParentInfo, 'groupParentInfo should be defined');
  const { theme } = useDesignSystemTheme();
  const location = useLocation();

  const { useGroupedValuesInCharts } = useExperimentViewRunsTableHeaderContext();
  const getRunColor = useGetExperimentRunColor();
  const saveRunColor = useSaveExperimentRunColor();
  const updateUIState = useUpdateExperimentViewUIState();
  const onExpandToggle = useCallback(
    (groupId: string, doOpen: boolean) => {
      updateUIState((current) => {
        const { groupsExpanded } = current;
        return {
          ...current,
          groupsExpanded: { ...groupsExpanded, [groupId]: doOpen },
        };
      });
    },
    [updateUIState],
  );

  const groupName = getRunGroupDisplayName(groupParentInfo);
  const groupIsDisplayedInCharts = useMemo(() => {
    if (shouldEnableToggleIndividualRunsInGroups()) {
      return useGroupedValuesInCharts && !isRemainingRunsGroup(groupParentInfo);
    }

    return !isRemainingRunsGroup(groupParentInfo);
  }, [groupParentInfo, useGroupedValuesInCharts]);

  const urlToRunUuidsFilter = useMemo(() => {
    const filter = createSearchFilterFromRunGroupInfo(groupParentInfo);

    const searchParams = new URLSearchParams(location.search);
    searchParams.set('searchFilter', filter);
    searchParams.set(EXPERIMENT_PAGE_QUERY_PARAM_IS_PREVIEW, 'true');
    const destination: To = {
      ...location,
      search: searchParams.toString(),
    };

    return destination;
  }, [groupParentInfo, location]);

  return (
    <div css={{ display: 'flex', gap: theme.spacing.sm, alignItems: 'center' }}>
      {groupParentInfo.expanderOpen ? (
        <ChevronDownIcon
          role="button"
          onClick={() => {
            onExpandToggle(groupParentInfo.groupId, false);
          }}
        />
      ) : (
        <ChevronRightIcon
          role="button"
          onClick={() => {
            onExpandToggle(groupParentInfo.groupId, true);
          }}
        />
      )}
      {/* Display color pill only when it's displayed in chart area */}
      {groupIsDisplayedInCharts && (
        <RunColorPill
          color={getRunColor(groupParentInfo.groupId)}
          hidden={isComparingRuns && hidden}
          onChangeColor={(colorValue) => {
            saveRunColor({ groupUuid: groupParentInfo.groupId, colorValue });
          }}
        />
      )}
      <div
        css={{
          display: 'inline-flex',
          gap: theme.spacing.sm,
          alignItems: 'center',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
      >
        {isRemainingRunsGroup(groupParentInfo) ? (
          <FormattedMessage
            defaultMessage="Additional runs"
            description="Experiment page > grouped runs table > label for group with additional, ungrouped runs"
          />
        ) : (
          <span title={groupName} css={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>
            <FormattedMessage
              defaultMessage="Group: {groupName}"
              description="Experiment page > grouped runs table > run group header label"
              values={{ groupName }}
            />
          </span>
        )}
        <Tag
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_groupparentcellrenderer.tsx_109"
          css={{ marginLeft: 0, marginRight: 0 }}
        >
          {groupParentInfo.runUuids.length}
        </Tag>
        <Tooltip
          componentId="codegen_mlflow_app_src_experiment-tracking_components_experiment-page_components_runs_cells_groupparentcellrenderer.tsx_136"
          content={
            <FormattedMessage
              defaultMessage="Open runs in this group in the new tab"
              description="Experiment page > grouped runs table > tooltip for a button that opens runs in a group in a new tab"
            />
          }
        >
          <Link
            to={urlToRunUuidsFilter}
            target="_blank"
            css={{
              marginLeft: -theme.spacing.xs,
              display: 'none',
              '.ag-row-hover &': {
                display: 'inline-flex',
              },
            }}
          >
            <Button
              type="link"
              componentId="mlflow.experiment_page.grouped_runs.open_runs_in_new_tab"
              size="small"
              icon={<NewWindowIcon css={{ svg: { width: 12, height: 12 } }} />}
            />
          </Link>
        </Tooltip>
      </div>
    </div>
  );
};
