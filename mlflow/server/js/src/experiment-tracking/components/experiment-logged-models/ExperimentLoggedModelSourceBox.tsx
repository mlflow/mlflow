import {
  BranchIcon,
  CopyIcon,
  GitCommitIcon,
  Tag,
  Typography,
  useDesignSystemTheme,
  Tooltip,
  Popover,
} from '@databricks/design-system';
import Utils from '../../../common/utils/Utils';
import type { LoggedModelKeyValueProto, LoggedModelProto } from '../../types';
import { MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG } from '../../constants';
import { CopyButton } from '@mlflow/mlflow/src/shared/building_blocks/CopyButton';
import { ExperimentSourceTypeIcon } from '../ExperimentSourceTypeIcon';
import { useMemo } from 'react';
import { useSearchParams } from '../../../common/utils/RoutingUtils';

export const ExperimentLoggedModelSourceBox = ({
  loggedModel,
  displayDetails,
  className,
}: {
  loggedModel: LoggedModelProto;
  /**
   * Set to true to display the branch name and commit hash.
   */
  displayDetails?: boolean;
  className?: string;
}) => {
  const [searchParams] = useSearchParams();

  const tagsByKey = useMemo(
    () =>
      loggedModel?.info?.tags?.reduce((acc, tag) => {
        if (!tag.key) {
          return acc;
        }
        acc[tag.key] = tag;
        return acc;
      }, {} as Record<string, LoggedModelKeyValueProto>) ?? {},
    [loggedModel?.info?.tags],
  );

  const branchName = tagsByKey?.[MLFLOW_RUN_GIT_SOURCE_BRANCH_TAG]?.value;
  const commitHash = tagsByKey?.[Utils.gitCommitTag]?.value;

  const runSource = useMemo(() => {
    try {
      return Utils.renderSource(tagsByKey, searchParams.toString(), undefined, branchName);
    } catch (e) {
      return undefined;
    }
  }, [tagsByKey, searchParams, branchName]);

  const sourceTypeValue = tagsByKey[Utils.sourceTypeTag]?.value;

  const { theme } = useDesignSystemTheme();
  return runSource ? (
    <div
      css={{
        display: 'flex',
        alignItems: 'center',
        gap: theme.spacing.sm,
        paddingTop: theme.spacing.sm,
        paddingBottom: theme.spacing.sm,
        flexWrap: displayDetails ? 'wrap' : undefined,
      }}
      className={className}
    >
      {sourceTypeValue && (
        <ExperimentSourceTypeIcon
          sourceType={sourceTypeValue}
          css={{ color: theme.colors.actionPrimaryBackgroundDefault }}
        />
      )}
      {runSource}{' '}
      {displayDetails && branchName && (
        <Tooltip componentId="mlflow.logged_model.details.source.branch_tooltip" content={branchName}>
          <Tag componentId="mlflow.logged_model.details.source.branch" css={{ marginRight: 0 }}>
            <div css={{ display: 'flex', gap: theme.spacing.xs, whiteSpace: 'nowrap' }}>
              <BranchIcon /> {branchName}
            </div>
          </Tag>
        </Tooltip>
      )}
      {displayDetails && commitHash && (
        <Popover.Root componentId="mlflow.logged_model.details.source.commit_hash_popover">
          <Popover.Trigger asChild>
            <Tag
              componentId="mlflow.logged_model.details.source.commit_hash"
              css={{ marginRight: 0, '&>div': { paddingRight: 0 } }}
            >
              <div css={{ display: 'flex', gap: theme.spacing.xs, whiteSpace: 'nowrap', alignContent: 'center' }}>
                <GitCommitIcon />
                {commitHash.slice(0, 7)}
              </div>
            </Tag>
          </Popover.Trigger>
          <Popover.Content align="start">
            <Popover.Arrow />
            <div css={{ display: 'flex', gap: theme.spacing.xs, alignItems: 'center' }}>
              {commitHash}
              <CopyButton showLabel={false} size="small" type="tertiary" copyText={commitHash} icon={<CopyIcon />} />
            </div>
          </Popover.Content>
        </Popover.Root>
      )}
    </div>
  ) : (
    <Typography.Hint>—</Typography.Hint>
  );
};
