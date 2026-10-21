import type { ReactElement, ReactNode } from 'react';
import { Button, Empty, NoIcon, PlusIcon, SearchIcon } from '@databricks/design-system';
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

/** `onCreateSkill` is set only when the whole catalog is empty, matching the MCP registry's empty state. */
export const SkillsEmptyState = ({
  isFiltered,
  onCreateSkill,
}: {
  isFiltered?: boolean;
  onCreateSkill?: () => void;
}) => {
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
  if (onCreateSkill) {
    return (
      <SkillRegistryEmptyState
        title={
          <FormattedMessage defaultMessage="Create skill" description="Empty state title for the Skill Registry" />
        }
        description={
          <FormattedMessage
            defaultMessage="Register and catalog skills for your organization."
            description="Empty state description for an empty Skill Registry"
          />
        }
        button={
          <Button
            componentId="mlflow.skill_registry.empty_state.create"
            type="primary"
            icon={<PlusIcon />}
            onClick={onCreateSkill}
          >
            <FormattedMessage defaultMessage="Create skill" description="Skill Registry empty state call to action" />
          </Button>
        }
      />
    );
  }
  return (
    <SkillRegistryEmptyState
      image={<NoIcon />}
      title={
        <FormattedMessage defaultMessage="No skills yet" description="Empty state title for a Skill Registry page" />
      }
      description={
        <FormattedMessage
          defaultMessage="Skills you can read will appear here once they are registered."
          description="Empty state description for a Skill Registry page without skills"
        />
      }
    />
  );
};
