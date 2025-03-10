import { FormattedMessage } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import { DetailsOverviewMetadataRow } from '../../../components/DetailsOverviewMetadataRow';
import { DetailsOverviewMetadataTable } from '../../../components/DetailsOverviewMetadataTable';
import { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { PromptsListTableTagsBox } from './PromptDetailsTagsBox';
import { PromptDetailsSourceRunsBox } from './PromptDetailsSourceRunsBox';
import { useMemo } from 'react';
import { REGISTERED_PROMPT_SOURCE_RUN_IDS } from '../utils';

export const PromptDetailsMetadata = ({
  promptEntity,
  onTagsUpdated,
}: {
  promptEntity?: RegisteredPrompt;
  onTagsUpdated?: () => void;
}) => {
  const sourceRunIds = useMemo(() => {
    const tagValue = promptEntity?.tags?.find((tag) => tag.key === REGISTERED_PROMPT_SOURCE_RUN_IDS)?.value;
    if (!tagValue) {
      return [];
    }
    return tagValue.split(',').map((runId) => runId.trim());
  }, [promptEntity]);

  return (
    <DetailsOverviewMetadataTable>
      <DetailsOverviewMetadataRow
        title={
          <FormattedMessage
            defaultMessage="Created at"
            description="Label for the creation time on the registered prompt details page"
          />
        }
        value={Utils.formatTimestamp(promptEntity?.creation_timestamp)}
      />
      <DetailsOverviewMetadataRow
        title={
          <FormattedMessage
            defaultMessage="Updated at"
            description="Label for the last update time on the registered prompt details page"
          />
        }
        value={Utils.formatTimestamp(promptEntity?.last_updated_timestamp)}
      />
      <DetailsOverviewMetadataRow
        title={
          <FormattedMessage
            defaultMessage="Tags"
            description="Label for the tags on the registered prompt details page"
          />
        }
        value={<PromptsListTableTagsBox onTagsUpdated={onTagsUpdated} promptEntity={promptEntity} />}
      />
      {sourceRunIds.length > 0 && (
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Source runs"
              description="Label for the related runs on the registered prompt details page"
            />
          }
          value={<PromptDetailsSourceRunsBox sourceRunIds={sourceRunIds} />}
        />
      )}
    </DetailsOverviewMetadataTable>
  );
};
