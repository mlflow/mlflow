import { ParagraphSkeleton, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ModelVersionTableAliasesCell } from '../../../../model-registry/components/aliases/ModelVersionTableAliasesCell';
import { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import Utils from '../../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { usePromptRunsInfo } from '../hooks/usePromptRunsInfo';
import { REGISTERED_PROMPT_SOURCE_RUN_IDS } from '../utils';
import { Fragment, useMemo } from 'react';

export const PromptVersionMetadata = ({
  registeredPromptVersion,
  registeredPrompt,
  showEditAliasesModal,
  onEditVersion,
  aliasesByVersion,
  isBaseline,
}: {
  registeredPrompt?: RegisteredPrompt;
  registeredPromptVersion?: RegisteredPromptVersion;
  showEditAliasesModal?: (versionNumber: string) => void;
  onEditVersion?: (vesrion: RegisteredPromptVersion) => void;
  aliasesByVersion: Record<string, string[]>;
  isBaseline?: boolean;
}) => {
  const { theme } = useDesignSystemTheme();


  const runIds = useMemo(() => {
    const tagValue = registeredPromptVersion?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_SOURCE_RUN_IDS)?.value;
    if (!tagValue) {
      return [];
    }
    return tagValue.split(',').map((runId) => runId.trim());
  }, [registeredPromptVersion]);


  const {
    isLoading: isLoadingRuns,
    runInfoMap: runInfoMap,
  } = usePromptRunsInfo(runIds ? runIds : []);

  if (!registeredPrompt || !registeredPromptVersion) {
    return null;
  }

  const versionElement = (
    <FormattedMessage
      defaultMessage="Version {version}"
      values={{ version: registeredPromptVersion.version }}
      description="A label for the version number in the prompt details page"
    />
  );

  return (
    <div
      css={{
        display: 'grid',
        gridTemplateColumns: '120px 1fr',
        gridAutoRows: `minmax(${theme.typography.lineHeightLg}, auto)`,
        alignItems: 'flex-start',
        rowGap: theme.spacing.xs,
        columnGap: theme.spacing.sm,
      }}
    >
      {onEditVersion && (
        <>
          <Typography.Text bold>Version:</Typography.Text>
          <Typography.Text>
            <Typography.Link
              componentId="mlflow.prompts.details.version.goto"
              onClick={() => onEditVersion(registeredPromptVersion)}
            >
              {versionElement}
            </Typography.Link>{' '}
            {isBaseline && (
              <FormattedMessage
                defaultMessage="(baseline)"
                description="A label displayed next to baseline version in the prompt versions comparison view"
              />
            )}
          </Typography.Text>
        </>
      )}
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="Registered at:"
          description="A label for the registration timestamp in the prompt details page"
        />
      </Typography.Text>
      <Typography.Text>{Utils.formatTimestamp(registeredPromptVersion.creation_timestamp)}</Typography.Text>
      <Typography.Text bold>
        <FormattedMessage
          defaultMessage="Aliases:"
          description="A label for the aliases list in the prompt details page"
        />
      </Typography.Text>
      <div>
        <ModelVersionTableAliasesCell
          css={{ maxWidth: 'none' }}
          modelName={registeredPrompt.name}
          version={registeredPromptVersion.version}
          aliases={aliasesByVersion[registeredPromptVersion.version] || []}
          onAddEdit={() => {
            showEditAliasesModal?.(registeredPromptVersion.version);
          }}
        />
      </div>
      {(isLoadingRuns || runIds) && (
        <>
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="MLflow runs:"
              description="A label for the associated MLflow runs in the prompt details page"
            />
          </Typography.Text>
          <Typography.Text>
            {isLoadingRuns ? (
              <ParagraphSkeleton css={{ width: 100 }} />
            ) : runIds.map((runId, index) => {
              const runInfo = runInfoMap[runId];
              const element = runInfo?.experimentId && runInfo?.runUuid && runInfo?.runName ? (
                <Link key={runId} to={Routes.getRunPageRoute(runInfo.experimentId, runInfo.runUuid)}>
                  {runInfo.runName}
                </Link>
              ) : (
                <span key={runId}>{runInfo?.runName || runInfo?.runUuid}</span>
              );

              // Add comma and space after each element except the last one
              return index < runIds.length - 1 ? (
                <Fragment key={`${runId}-fragment`}>
                  {element},{' '}
                </Fragment>
              ) : element;
            })}
          </Typography.Text>
        </>
      )}
      {registeredPromptVersion.description && (
        <>
          <Typography.Text bold>
            <FormattedMessage
              defaultMessage="Commit message:"
              description="A label for the commit message in the prompt details page"
            />
          </Typography.Text>
          <Typography.Text>{registeredPromptVersion.description}</Typography.Text>
        </>
      )}
    </div>
  );
};
