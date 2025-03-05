import { FormattedMessage } from 'react-intl';
import Utils from '../../../../common/utils/Utils';
import { DetailsOverviewMetadataRow } from '../../../components/DetailsOverviewMetadataRow';
import { DetailsOverviewMetadataTable } from '../../../components/DetailsOverviewMetadataTable';
import { RegisteredPrompt } from '../types';
import { PromptsListTableTagsBox } from './PromptDetailsTagsBox';

export const PromptDetailsMetadata = ({
  promptEntity,
  onTagsUpdated,
}: {
  promptEntity?: RegisteredPrompt;
  onTagsUpdated?: () => void;
}) => {
  return (
    <DetailsOverviewMetadataTable>
      <DetailsOverviewMetadataRow
        title={<FormattedMessage defaultMessage="Created at" description="TODO" />}
        value={Utils.formatTimestamp(promptEntity?.creation_timestamp)}
      />
      <DetailsOverviewMetadataRow
        title={<FormattedMessage defaultMessage="Updated at" description="TODO" />}
        value={Utils.formatTimestamp(promptEntity?.last_updated_timestamp)}
      />
      <DetailsOverviewMetadataRow
        title={<FormattedMessage defaultMessage="Tags" description="TODO" />}
        value={<PromptsListTableTagsBox onTagsUpdated={onTagsUpdated} promptEntity={promptEntity} />}
      />
    </DetailsOverviewMetadataTable>
  );
};
