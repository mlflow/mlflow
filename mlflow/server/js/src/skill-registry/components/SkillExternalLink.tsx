import { NewWindowIcon, Typography } from '@databricks/design-system';

// Long URLs wrap anywhere so they stay inside their container, with the icon following the last line.
export const SkillExternalLink = ({ componentId, href }: { componentId: string; href: string }) => (
  <Typography.Link
    componentId={componentId}
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    css={{ maxWidth: '100%', overflowWrap: 'anywhere' }}
  >
    {href}
    <span aria-hidden css={{ marginLeft: 4, whiteSpace: 'nowrap' }}>
      <NewWindowIcon css={{ fontSize: 12, verticalAlign: 'middle' }} />
    </span>
  </Typography.Link>
);
