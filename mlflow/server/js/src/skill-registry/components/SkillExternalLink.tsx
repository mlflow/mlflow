import { NewWindowIcon, Typography } from '@databricks/design-system';

export const SkillExternalLink = ({ componentId, href }: { componentId: string; href: string }) => (
  <Typography.Link componentId={componentId} href={href} target="_blank" rel="noopener noreferrer">
    <span css={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      {href}
      <span aria-hidden>
        <NewWindowIcon css={{ fontSize: 12 }} />
      </span>
    </span>
  </Typography.Link>
);
