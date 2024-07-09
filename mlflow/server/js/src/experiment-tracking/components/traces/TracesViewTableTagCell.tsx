import { Button, PencilIcon, SpeechBubblePlusIcon, useDesignSystemTheme } from '@databricks/design-system';
import { MLFLOW_INTERNAL_PREFIX } from '../../../common/utils/TagUtils';
import { KeyValueTag } from '../../../common/components/KeyValueTag';

export const TracesViewTableTagCell = ({
  onAddEditTags,
  tags,
}: {
  tags: { key: string; value: string }[];
  onAddEditTags: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const visibleTagList = tags?.filter(({ key }) => key && !key.startsWith(MLFLOW_INTERNAL_PREFIX)) || [];
  const containsTags = visibleTagList.length > 0;
  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        flexWrap: 'wrap',
        columnGap: theme.spacing.xs / 2,
        rowGap: theme.spacing.xs / 2,
      }}
    >
      {visibleTagList.map((tag) => (
        <KeyValueTag
          key={tag.key}
          tag={tag}
          css={{ marginRight: 0 }}
          charLimit={20}
          maxWidth={150}
          enableFullViewModal
        />
      ))}{' '}
      <Button
        componentId="mlflow.experiment_page.traces_table.edit_tag"
        size="small"
        icon={!containsTags ? <SpeechBubblePlusIcon /> : <PencilIcon />}
        onClick={onAddEditTags}
      />
    </div>
  );
};
