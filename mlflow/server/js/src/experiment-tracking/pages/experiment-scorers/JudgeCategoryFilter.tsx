import React from 'react';
import { useDesignSystemTheme, Input, SearchIcon, Button } from '@databricks/design-system';
import { FormattedMessage, useIntl } from '@databricks/i18n';
import { JudgeCategory, JUDGE_CATEGORIES } from './types';
import { COMPONENT_ID_PREFIX } from './constants';

interface JudgeCategoryFilterProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  activeCategory: JudgeCategory | 'all';
  onCategoryChange: (category: JudgeCategory | 'all') => void;
}

const CATEGORY_LABELS: Record<JudgeCategory | 'all', React.ReactNode> = {
  all: <FormattedMessage defaultMessage="All" description="Filter label for all judge categories" />,
  [JudgeCategory.RAG]: <FormattedMessage defaultMessage="RAG" description="Filter label for RAG judges" />,
  [JudgeCategory.TEXT_QUALITY]: (
    <FormattedMessage defaultMessage="Text Quality" description="Filter label for text quality judges" />
  ),
  [JudgeCategory.SAFETY]: (
    <FormattedMessage defaultMessage="Safety" description="Filter label for safety judges" />
  ),
  [JudgeCategory.TOOL_CALL]: (
    <FormattedMessage defaultMessage="Tool Call" description="Filter label for tool call judges" />
  ),
  [JudgeCategory.AGENT]: <FormattedMessage defaultMessage="Agent" description="Filter label for agent judges" />,
};

const JudgeCategoryFilter: React.FC<JudgeCategoryFilterProps> = ({
  searchQuery,
  onSearchChange,
  activeCategory,
  onCategoryChange,
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();

  const categories: (JudgeCategory | 'all')[] = ['all', ...JUDGE_CATEGORIES];

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.sm }}>
      <Input
        componentId={`${COMPONENT_ID_PREFIX}.category-filter-search`}
        prefix={<SearchIcon />}
        value={searchQuery}
        onChange={(e) => onSearchChange(e.target.value)}
        placeholder={intl.formatMessage({
          defaultMessage: 'Search judges by name...',
          description: 'Placeholder for judge search input',
        })}
        allowClear
        css={{ width: '100%' }}
      />
      <div css={{ display: 'flex', gap: theme.spacing.xs, flexWrap: 'wrap' }}>
        {categories.map((category) => {
          const isActive = activeCategory === category;
          return (
            <Button
              key={category}
              componentId={`${COMPONENT_ID_PREFIX}.category-chip-${category}`}
              size="small"
              type={isActive ? 'primary' : 'tertiary'}
              onClick={() => onCategoryChange(category)}
              css={{
                borderRadius: theme.spacing.md,
              }}
            >
              {CATEGORY_LABELS[category]}
            </Button>
          );
        })}
      </div>
    </div>
  );
};

export default JudgeCategoryFilter;
