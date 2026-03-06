import React from 'react';
import {
  useDesignSystemTheme,
  Typography,
  Checkbox,
  SearchIcon,
  ThumbsDownIcon,
  ShieldIcon,
  LightningIcon,
} from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';

export type IssueCategory = 'low_quality' | 'negative_ux' | 'safety' | 'performance';

interface IssueCategoryDefinition {
  id: IssueCategory;
  icon: React.ReactNode;
  title: React.ReactNode;
  description: React.ReactNode;
}

export const ISSUE_CATEGORY_DEFINITIONS: IssueCategoryDefinition[] = [
  {
    id: 'low_quality',
    icon: <SearchIcon />,
    title: (
      <FormattedMessage
        defaultMessage="Low Quality Responses"
        description="Issue category title for detecting low quality responses"
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Detect incorrect, incomplete, or hallucinated responses"
        description="Issue category description for low quality responses"
      />
    ),
  },
  {
    id: 'negative_ux',
    icon: <ThumbsDownIcon />,
    title: (
      <FormattedMessage
        defaultMessage="Negative User Experiences"
        description="Issue category title for detecting negative user experiences"
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Detect unsatisfactory user experiences and unmet user needs"
        description="Issue category description for negative user experiences"
      />
    ),
  },
  {
    id: 'safety',
    icon: <ShieldIcon />,
    title: (
      <FormattedMessage
        defaultMessage="Safety & Compliance Violations"
        description="Issue category title for detecting safety violations"
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Detect unsafe responses or actions and compliance violations"
        description="Issue category description for safety violations"
      />
    ),
  },
  {
    id: 'performance',
    icon: <LightningIcon />,
    title: (
      <FormattedMessage
        defaultMessage="Performance Issues"
        description="Issue category title for detecting performance issues"
      />
    ),
    description: (
      <FormattedMessage
        defaultMessage="Identify slow requests or operations"
        description="Issue category description for performance issues"
      />
    ),
  },
];

export const ALL_ISSUE_CATEGORIES: IssueCategory[] = ISSUE_CATEGORY_DEFINITIONS.map((def) => def.id);

interface IssueCategoryCardProps {
  category: IssueCategoryDefinition;
  isSelected: boolean;
  onToggle: (categoryId: IssueCategory, isChecked: boolean) => void;
}

const IssueCategoryCard: React.FC<IssueCategoryCardProps> = ({ category, isSelected, onToggle }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div
      css={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: theme.spacing.md,
        padding: theme.spacing.md,
        border: `1px solid ${theme.colors.border}`,
        borderRadius: theme.borders.borderRadiusMd,
        cursor: 'pointer',
        transition: 'background-color 0.2s',
        '&:hover': {
          backgroundColor: theme.colors.actionTertiaryBackgroundHover,
        },
      }}
      onClick={() => onToggle(category.id, !isSelected)}
    >
      <Checkbox
        componentId={`mlflow.traces.issue-detection-modal.category.${category.id}`}
        isChecked={isSelected}
        onChange={(checked) => onToggle(category.id, checked)}
        onClick={(e) => e.stopPropagation()}
      />
      <div css={{ color: theme.colors.textSecondary, marginTop: 2 }}>{category.icon}</div>
      <div css={{ flex: 1 }}>
        <Typography.Text css={{ fontWeight: theme.typography.typographyBoldFontWeight }}>
          {category.title}
        </Typography.Text>
        <Typography.Text color="secondary" css={{ display: 'block', marginTop: theme.spacing.xs }}>
          {category.description}
        </Typography.Text>
      </div>
    </div>
  );
};

interface IssueCategoryListProps {
  selectedCategories: Set<IssueCategory>;
  onToggle: (categoryId: IssueCategory, isChecked: boolean) => void;
}

export const IssueCategoryList: React.FC<IssueCategoryListProps> = ({ selectedCategories, onToggle }) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      {ISSUE_CATEGORY_DEFINITIONS.map((category) => (
        <IssueCategoryCard
          key={category.id}
          category={category}
          isSelected={selectedCategories.has(category.id)}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
};
