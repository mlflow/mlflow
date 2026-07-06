import { Typography } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

interface DatasetRecordDetailHeaderProps {
  recordId: string;
}

export const DatasetRecordDetailHeader = ({ recordId }: DatasetRecordDetailHeaderProps) => (
  <Typography.Text bold ellipsis>
    <FormattedMessage
      defaultMessage="Record {id}"
      description="Heading at the top of the V2 dataset record side panel showing the record id"
      values={{ id: recordId }}
    />
  </Typography.Text>
);
