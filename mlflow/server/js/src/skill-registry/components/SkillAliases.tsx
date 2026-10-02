import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import { AliasTag } from '../../common/components/AliasTag';
import { tagListStyles } from '../styles';

export const SkillAliases = ({ aliases }: { aliases: string[] }) => {
  const { theme } = useDesignSystemTheme();
  if (aliases.length === 0) {
    return <span aria-label="No aliases">—</span>;
  }
  return (
    <div css={tagListStyles(theme)}>
      {aliases.map((alias) => (
        <AliasTag value={alias} key={alias} />
      ))}
    </div>
  );
};
