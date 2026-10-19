import { Typography } from '@databricks/design-system';
import type { ReactNode } from 'react';

export interface IssueDetailSectionTitleProps {
  children: ReactNode;
  id?: string;
}

/** The title of a section in an issue's detail, set apart from the gray text under it. */
export const IssueDetailSectionTitle = ({ children, id }: IssueDetailSectionTitleProps) => (
  <Typography.Title level={4} withoutMargins id={id}>
    {children}
  </Typography.Title>
);
