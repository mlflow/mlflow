import { FormattedMessage } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import { DetailsOverviewMetadataRow } from '../../../components/DetailsOverviewMetadataRow';
import { DetailsOverviewMetadataTable } from '../../../components/DetailsOverviewMetadataTable';
import { RegisteredPrompt, RegisteredPromptVersion } from '../types';
import { PromptsListTableTagsBox } from './PromptDetailsTagsBox';
import { PromptDetailsSourceRunsBox } from './PromptDetailsSourceRunsBox';
import { useMemo } from 'react';
import { REGISTERED_PROMPT_SOURCE_RUN_ID } from '../utils';

export const PromptDetailsMetadata = ({
  promptEntity,
  promptVersions = [],
  onTagsUpdated,
}: {
  promptEntity?: RegisteredPrompt;
  promptVersions?: RegisteredPromptVersion[];
  onTagsUpdated?: () => void;
}) => {
  const containsSourceIds = useMemo(
    () => promptVersions.some((version) => version.tags?.find((tag) => tag.key === REGISTERED_PROMPT_SOURCE_RUN_ID)),
    [promptVersions],
  );

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
      {containsSourceIds && (
        <DetailsOverviewMetadataRow
          title={
            <FormattedMessage
              defaultMessage="Source runs"
              description="Label for the related runs on the registered prompt details page"
            />
          }
          value={<PromptDetailsSourceRunsBox promptVersions={promptVersions} />}
        />
      )}
    </DetailsOverviewMetadataTable>
  );
};
