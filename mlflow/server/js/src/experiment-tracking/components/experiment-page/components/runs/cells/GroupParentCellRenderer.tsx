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
import { isRemainingRunsGroup } from '../../../utils/experimentPage.group-row-utils';
import { useUpdateExperimentViewUIState } from '../../../contexts/ExperimentPageUIStateContext';
import { useCallback } from 'react';
import { RunColorPill } from '../../RunColorPill';
import { isObject } from 'lodash';
import invariant from 'invariant';
import { FormattedMessage } from 'react-intl';

export interface GroupParentCellRendererProps extends ICellRendererParams {
  data: RunRowType;
}

export const GroupParentCellRenderer = ({ data }: GroupParentCellRendererProps) => {
  const groupParentInfo = data.groupParentInfo;
  invariant(groupParentInfo, 'groupParentInfo should be defined');
  const { theme } = useDesignSystemTheme();

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

  const isGroupingByDataset = groupParentInfo.groupingMode === RunGroupingMode.Dataset;

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
      }}
    >
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
        {/* Display color pill only for non-remaining runs groups */}
        {!isRemainingRunsGroup(groupParentInfo) && <RunColorPill color={data.color} />}
        <div
          css={{
            display: 'flex',
            flexShrink: 0,
            gap: theme.spacing.sm,
            alignItems: 'center',
            overflow: 'hidden',
            marginRight: theme.spacing.sm,
          }}
        >
          {isRemainingRunsGroup(groupParentInfo) ? (
            <FormattedMessage
              defaultMessage="Additional runs"
              description="Experiment page > grouped runs table > label for group with additional, ungrouped runs"
            />
          ) : (
            <>
              {isGroupingByDataset && isObject(groupParentInfo.value) ? (
                <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.xs }}>
                  <TableIcon css={{ color: theme.colors.textSecondary, marginRight: theme.spacing.xs }} />
                  {groupParentInfo.value.name} ({groupParentInfo.value.digest})
                </div>
              ) : (
                <FormattedMessage
                  defaultMessage="Group: {groupName}"
                  description="Experiment page > grouped runs table > run group header label"
                  values={{ groupName: groupParentInfo.value }}
                />
              )}
            </>
          )}
          <Tag css={{ marginLeft: 0 }}>{groupParentInfo.runUuids.length}</Tag>
        </div>
      </div>
    </div>
  );
};
