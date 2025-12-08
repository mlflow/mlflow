import { Button, ParagraphSkeleton, PencilIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { ModelVersionTableAliasesCell } from '../../../../model-registry/components/aliases/ModelVersionTableAliasesCell';
import type { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import Utils from '../../../../common/utils/Utils';
import { FormattedMessage } from 'react-intl';
import { Link } from '../../../../common/utils/RoutingUtils';
import Routes from '../../../routes';
import { usePromptRunsInfo } from '../hooks/usePromptRunsInfo';
import { getModelConfigFromTags, REGISTERED_PROMPT_SOURCE_RUN_IDS } from '../utils';
import { useCallback, useMemo } from 'react';
import { PromptVersionRuns } from './PromptVersionRuns';
import { isUserFacingTag } from '@mlflow/mlflow/src/common/utils/TagUtils';
import { KeyValueTag } from '@mlflow/mlflow/src/common/components/KeyValueTag';
import { PromptVersionTags } from './PromptVersionTags';

const MAX_VISIBLE_TAGS = 3;

export const PromptVersionMetadata = ({
  registeredPromptVersion,
  registeredPrompt,
  showEditAliasesModal,
  onEditVersion,
  showEditPromptVersionMetadataModal,
  showEditModelConfigModal,
  aliasesByVersion,
  isBaseline,
}: {
  registeredPrompt?: RegisteredPrompt;
  registeredPromptVersion?: RegisteredPromptVersion;
  showEditAliasesModal?: (versionNumber: string) => void;
  onEditVersion?: (vesrion: RegisteredPromptVersion) => void;
  showEditPromptVersionMetadataModal?: (version: RegisteredPromptVersion) => void;
  showEditModelConfigModal?: (version: RegisteredPromptVersion) => void;
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

  const { isLoading: isLoadingRuns, runInfoMap } = usePromptRunsInfo(runIds ? runIds : []);

  if (!registeredPrompt || !registeredPromptVersion) {
    return null;
  }

  const visibleTagList = registeredPromptVersion?.tags?.filter((tag) => isUserFacingTag(tag.key)) || [];

  const versionElement = (
    <FormattedMessage
      defaultMessage="Version {version}"
      values={{ version: registeredPromptVersion.version }}
      description="A label for the version number in the prompt details page"
    />
  );

  const onEditVersionMetadata = showEditPromptVersionMetadataModal
    ? () => {
        showEditPromptVersionMetadataModal(registeredPromptVersion);
      }
    : undefined;

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
      <PromptVersionTags onEditVersionMetadata={onEditVersionMetadata} tags={visibleTagList} />
      {/* Model Config Section */}
      {(() => {
        const modelConfig = getModelConfigFromTags(registeredPromptVersion?.tags);
        const hasModelConfig = !!modelConfig;

        // Only show section if there's config or ability to edit
        if (!hasModelConfig && !showEditModelConfigModal) return null;

        return (
          <>
            <Typography.Text bold>
              <FormattedMessage
                defaultMessage="Model Config:"
                description="Label for model configuration in the prompt details page"
              />
            </Typography.Text>
            <div css={{ display: 'flex', alignItems: 'flex-start', gap: theme.spacing.sm }}>
              {hasModelConfig ? (
                <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs, flex: 1 }}>
                  {modelConfig.model_name && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Model: </Typography.Text>
                      <Typography.Text>{modelConfig.model_name}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.temperature !== undefined && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Temperature: </Typography.Text>
                      <Typography.Text>{modelConfig.temperature}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.max_tokens !== undefined && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Max Tokens: </Typography.Text>
                      <Typography.Text>{modelConfig.max_tokens}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.top_p !== undefined && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Top P: </Typography.Text>
                      <Typography.Text>{modelConfig.top_p}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.top_k !== undefined && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Top K: </Typography.Text>
                      <Typography.Text>{modelConfig.top_k}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.frequency_penalty !== undefined && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Frequency Penalty: </Typography.Text>
                      <Typography.Text>{modelConfig.frequency_penalty}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.presence_penalty !== undefined && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Presence Penalty: </Typography.Text>
                      <Typography.Text>{modelConfig.presence_penalty}</Typography.Text>
                    </div>
                  )}
                  {modelConfig.stop_sequences && modelConfig.stop_sequences.length > 0 && (
                    <div>
                      <Typography.Text css={{ color: theme.colors.textSecondary }}>Stop Sequences: </Typography.Text>
                      <Typography.Text>{modelConfig.stop_sequences.join(', ')}</Typography.Text>
                    </div>
                  )}
                </div>
              ) : (
                <Typography.Hint>—</Typography.Hint>
              )}
              {showEditModelConfigModal && (
                <Button
                  componentId="mlflow.prompts.details.version.edit_model_config"
                  size="small"
                  icon={<PencilIcon />}
                  onClick={() => showEditModelConfigModal(registeredPromptVersion)}
                />
              )}
            </div>
          </>
        );
      })()}
      {(isLoadingRuns || runIds.length > 0) && (
        <PromptVersionRuns isLoadingRuns={isLoadingRuns} runIds={runIds} runInfoMap={runInfoMap} />
      )}
    </div>
  );
};
