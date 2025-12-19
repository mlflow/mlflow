import { Button, ChevronDownIcon, ChevronRightIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { Link } from '../../../common/utils/RoutingUtils';
import Routes from '../../routes';
import { useExperimentLoggedModelListPageTableContext } from './ExperimentLoggedModelListPageTableContext';
import type { LoggedModelDataGroupDataRow } from './ExperimentLoggedModelListPageTable.utils';
import { LoggedModelsTableSpecialRowID } from './ExperimentLoggedModelListPageTable.utils';
import { FormattedMessage } from 'react-intl';

export const ExperimentLoggedModelTableGroupCell = ({ data }: { data: LoggedModelDataGroupDataRow }) => {
  const { theme } = useDesignSystemTheme();
  const { expandedGroups, onGroupToggle } = useExperimentLoggedModelListPageTableContext();

  const groupId = data.groupUuid;
  const isExpanded = expandedGroups?.includes(groupId);

  return (
    <div css={{ display: 'flex', alignItems: 'center', gap: theme.spacing.sm }}>
      <Button
        icon={isExpanded ? <ChevronDownIcon /> : <ChevronRightIcon />}
        componentId="mlflow.logged_model_table.group_toggle"
        onClick={() => onGroupToggle?.(groupId)}
        size="small"
      />
      {data.groupData?.sourceRun ? (
        <Link
          to={Routes.getRunPageRoute(data.groupData.sourceRun.info.experimentId, data.groupData.sourceRun.info.runUuid)}
          target="_blank"
        >
          {data.groupData.sourceRun.info.runName || data.groupData.sourceRun.info.runUuid}
        </Link>
      ) : null}
      {groupId === LoggedModelsTableSpecialRowID.REMAINING_MODELS_GROUP ? (
        <Typography.Text>
          {/* Shouldn't really happen, but we should handle it gracefully */}
          <FormattedMessage
            defaultMessage="Ungrouped"
            description="Label for the group of logged models that are not grouped by any source run"
          />
        </Typography.Text>
      ) : null}
    </div>
  );
};
