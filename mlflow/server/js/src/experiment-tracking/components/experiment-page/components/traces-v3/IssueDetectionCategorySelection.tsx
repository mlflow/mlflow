import React from 'react';
import { useDesignSystemTheme, Typography } from '@databricks/design-system';
import { FormattedMessage } from '@databricks/i18n';
import { IssueCategoryList, type IssueCategory } from './IssueDetectionCategories';

interface IssueDetectionCategorySelectionProps {
  selectedCategories: Set<IssueCategory>;
  onCategoryToggle: (categoryId: IssueCategory, isChecked: boolean) => void;
}

export const IssueDetectionCategorySelection: React.FC<IssueDetectionCategorySelectionProps> = ({
  selectedCategories,
  onCategoryToggle,
}) => {
  const { theme } = useDesignSystemTheme();

  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.md }}>
      <div>
        <Typography.Title level={4} css={{ marginBottom: theme.spacing.xs }}>
          <FormattedMessage
            defaultMessage="Select Categories"
            description="Header for the category selection step in issue detection modal"
          />
        </Typography.Title>
        <Typography.Text color="secondary">
          <FormattedMessage
            defaultMessage="Choose which types of issues to detect in your traces"
            description="Description for the category selection step"
          />
        </Typography.Text>
      </div>
      <IssueCategoryList selectedCategories={selectedCategories} onToggle={onCategoryToggle} />
    </div>
  );
};
