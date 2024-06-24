import { ICellRendererParams } from '@ag-grid-community/core';
import { RunGroupParentInfo, RunGroupingMode, RunRowType } from '../../../utils/experimentPage.row-types';
import {
  Button,
  ChevronDownIcon,
  ChevronRightIcon,
  TableIcon,
  Tag,
  useDesignSystemTheme,
} from '@databricks/design-system';
import { getRunGroupDisplayName, isRemainingRunsGroup } from '../../../utils/experimentPage.group-row-utils';
import { useUpdateExperimentViewUIState } from '../../../contexts/ExperimentPageUIStateContext';
import { useCallback, useMemo } from 'react';
import { RunColorPill } from '../../RunColorPill';
import { isObject } from 'lodash';
import invariant from 'invariant';
import { FormattedMessage } from 'react-intl';
import { useGetExperimentRunColor, useSaveExperimentRunColor } from '../../../hooks/useExperimentRunColor';
import { useExperimentViewRunsTableHeaderContext } from '../ExperimentViewRunsTableHeaderContext';
import { shouldEnableToggleIndividualRunsInGroups } from '../../../../../../common/utils/FeatureUtils';

export interface GroupParentCellRendererProps extends ICellRendererParams {
  data: RunRowType;
  isComparingRuns?: boolean;
}

export const GroupParentCellRenderer = ({ data, isComparingRuns }: GroupParentCellRendererProps) => {
  const groupParentInfo = data.groupParentInfo;
  const hidden = data.hidden;
  invariant(groupParentInfo, 'groupParentInfo should be defined');
  const { theme } = useDesignSystemTheme();

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
        <Tag css={{ marginLeft: 0, marginRight: 0 }}>{groupParentInfo.runUuids.length}</Tag>
      </div>
    </div>
  );
};
