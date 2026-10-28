import { useDesignSystemTheme } from '@databricks/design-system';
import { useIntl } from 'react-intl';
import { AliasTag } from '../../common/components/AliasTag';
import { tagListStyles } from '../styles';
import { SkillLatestAliasTag } from './SkillLatestAliasTag';
import { SkillPencilButton } from './SkillPencilButton';

export const SkillAliases = ({
  aliases,
  isLatest = false,
  onEdit,
}: {
  aliases: string[];
  isLatest?: boolean;
  onEdit?: () => void;
}) => {
  const { theme } = useDesignSystemTheme();
  const intl = useIntl();
  const content =
    aliases.length === 0 && !isLatest ? (
      <span
        aria-label={intl.formatMessage({
          defaultMessage: 'No aliases',
          description: 'Accessible label for an empty skill alias list',
        })}
      >
        —
      </span>
    ) : (
      <>
        {isLatest && <SkillLatestAliasTag />}
        {aliases.map((alias) => (
          <AliasTag value={alias} key={alias} />
        ))}
      </>
    );
  return (
    <div css={tagListStyles(theme)}>
      {content}
      {onEdit && (
        <SkillPencilButton
          componentId="mlflow.skill_registry.detail.version.aliases.edit"
          label={intl.formatMessage({
            defaultMessage: 'Edit aliases',
            description: 'Aria label for editing skill version aliases',
          })}
          onClick={onEdit}
        />
      )}
    </div>
  );
};
