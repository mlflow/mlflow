import type { CursorPaginationProps } from '@databricks/design-system';
import { CursorPagination, Spinner, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import type { Skill } from '../types';
import { SkillCard } from './SkillCard';
import { SkillsEmptyState } from './SkillRegistryEmptyState';
import { cardGridStyles, flexColumnContainerStyles } from '../styles';

export const SkillCardGrid = ({
  skills,
  isLoading,
  isFiltered,
  onCreateSkill,
  hasNextPage,
  hasPreviousPage,
  onNextPage,
  onPreviousPage,
  pageSizeSelect,
}: {
  skills?: Skill[];
  isLoading?: boolean;
  isFiltered?: boolean;
  onCreateSkill?: () => void;
  hasNextPage: boolean;
  hasPreviousPage: boolean;
  onNextPage: () => void;
  onPreviousPage: () => void;
  pageSizeSelect?: CursorPaginationProps['pageSizeSelect'];
}) => {
  const { theme } = useDesignSystemTheme();

  if (isLoading) {
    return (
      <div
        role="status"
        css={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: theme.spacing.sm,
          padding: theme.spacing.lg,
          minHeight: 200,
        }}
      >
        <Spinner size="small" />
        <FormattedMessage defaultMessage="Loading skills..." description="Loading state for Skill Registry card grid" />
      </div>
    );
  }

  return (
    <div css={{ ...flexColumnContainerStyles, minHeight: 0 }}>
      {skills?.length ? (
        <div role="list" aria-label="Skills" css={cardGridStyles(theme)}>
          {skills.map((skill) => (
            <div role="listitem" key={formatSkillListKey(skill)}>
              <SkillCard skill={skill} />
            </div>
          ))}
        </div>
      ) : (
        <SkillsEmptyState isFiltered={isFiltered} onCreateSkill={onCreateSkill} />
      )}
      {(skills?.length || hasNextPage || hasPreviousPage) && (
        <div
          css={{
            flexShrink: 0,
            display: 'flex',
            justifyContent: 'flex-end',
            paddingTop: theme.spacing.sm,
            paddingBottom: theme.spacing.sm,
          }}
        >
          <CursorPagination
            hasNextPage={hasNextPage}
            hasPreviousPage={hasPreviousPage}
            onNextPage={onNextPage}
            onPreviousPage={onPreviousPage}
            pageSizeSelect={pageSizeSelect}
            componentId="mlflow.skill_registry.grid.pagination"
          />
        </div>
      )}
    </div>
  );
};

const formatSkillListKey = (skill: { name: string; organization: string }) =>
  skill.organization ? `@${skill.organization}/${skill.name}` : skill.name;
