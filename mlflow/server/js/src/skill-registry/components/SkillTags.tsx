import { Overflow, useDesignSystemTheme } from '@databricks/design-system';
import { KeyValueTag } from '../../common/components/KeyValueTag';
import { inlineFlexRowStyles, tagListStyles } from '../styles';

export const SkillTags = ({ tags, wrap = false }: { tags: Record<string, string>; wrap?: boolean }) => {
  const { theme } = useDesignSystemTheme();
  const entries = Object.entries(tags);
  if (entries.length === 0) return <span aria-label="No tags">—</span>;
  const tagNodes = entries.map(([key, value]) => <KeyValueTag key={key} css={{ margin: 0 }} tag={{ key, value }} />);
  return (
    <div
      css={wrap ? tagListStyles(theme) : inlineFlexRowStyles(theme)}
      onClick={(e) => {
        if ((e.target as HTMLElement).closest('button')) {
          e.stopPropagation();
        }
      }}
    >
      {wrap ? tagNodes : <Overflow>{tagNodes}</Overflow>}
    </div>
  );
};
