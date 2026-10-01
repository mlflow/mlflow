import type { ReactElement, ReactNode } from 'react';
import { Empty, NoIcon, SearchIcon } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { emptyCenterStyles } from '../styles';

export const SkillRegistryEmptyState = ({
  title,
  description,
  button,
  image,
}: {
  title: ReactNode;
  description?: ReactNode;
  button?: ReactElement;
  image?: ReactElement;
}) => {
  return (
    <div css={emptyCenterStyles}>
      <Empty title={title} description={description ?? null} button={button} image={image} />
    </div>
  );
};

export const SkillsEmptyState = ({ isFiltered }: { isFiltered?: boolean }) => {
  if (isFiltered) {
    return (
      <SkillRegistryEmptyState
        image={<SearchIcon />}
        title={
          <FormattedMessage
            defaultMessage="No skills found"
            description="Empty state when skill search returns no results"
          />
        }
        description={
          <FormattedMessage
            defaultMessage="Try clearing your filters or searching for a different term."
            description="Empty-search description for the Skill Registry catalog"
          />
        }
      />
    );
  }
  return (
    <SkillRegistryEmptyState
      image={<NoIcon />}
      title={<FormattedMessage defaultMessage="No skills yet" description="Empty state title for the Skill Registry" />}
      description={
        <FormattedMessage
          defaultMessage="Skills you can read will appear here once they are registered."
          description="Empty state description for an empty Skill Registry"
        />
      }
    />
  );
};
