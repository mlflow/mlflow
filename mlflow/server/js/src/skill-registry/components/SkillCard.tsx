import { Card, Tooltip, Typography, useDesignSystemTheme } from '@databricks/design-system';

import type { Skill } from '../types';
import SkillRegistryRoutes from '../routes';
import { textClampStyles, textEllipsisStyles, cardBodyStyles, cardHeaderRowStyles, noShrinkStyles } from '../styles';
import { formatSkillOrganization, isSkillDimmed } from '../utils';
import { SkillIcon } from './SkillIcon';
import { SkillTags } from './SkillTags';
import { UseSkillButton } from './UseSkillButton';
import { useNavigate } from '../../common/utils/RoutingUtils';

export const SkillCard = ({ skill }: { skill: Skill }) => {
  const { theme } = useDesignSystemTheme();
  const navigate = useNavigate();
  const isDimmed = isSkillDimmed(skill);
  const hasTags = Object.keys(skill.tags || {}).length > 0;
  const organizationLabel = formatSkillOrganization(skill.organization);

  return (
    <Card
      componentId="mlflow.skill_registry.card"
      width="100%"
      navigateFn={async () => {
        navigate(SkillRegistryRoutes.getSkillDetailRoute(skill.name, skill.organization));
      }}
      disableHover={isDimmed}
      dangerouslyAppendEmotionCSS={{
        height: '100%',
        '& > div': { display: 'flex', flexDirection: 'column', flexGrow: 1 },
        ...(isDimmed
          ? {
              cursor: 'pointer',
              '&:hover': {
                borderColor: theme.colors.textSecondary,
              },
            }
          : {}),
      }}
    >
      <div css={{ ...cardBodyStyles(theme), opacity: isDimmed ? 0.5 : 1 }}>
        <div css={{ ...cardHeaderRowStyles(theme), minWidth: 0 }}>
          <SkillIcon icons={skill.icons} name={skill.name} />
          <Tooltip content={skill.name} componentId="mlflow.skill_registry.card.name_tooltip">
            <Typography.Text bold css={{ ...textEllipsisStyles, flex: 1, minWidth: 0 }}>
              {skill.name}
            </Typography.Text>
          </Tooltip>
          {skill.latest_version != null && (
            <Typography.Text color="secondary" size="sm" css={noShrinkStyles}>
              v{skill.latest_version}
            </Typography.Text>
          )}
        </div>
        {skill.description && (
          <Typography.Text color="secondary" size="sm" css={textClampStyles(hasTags ? 2 : 3)}>
            {skill.description}
          </Typography.Text>
        )}
        {hasTags && <SkillTags tags={skill.tags || {}} />}
        <div css={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 'auto' }}>
          {organizationLabel ? (
            <Typography.Text color="secondary" size="sm" css={textEllipsisStyles}>
              {organizationLabel}
            </Typography.Text>
          ) : (
            <span />
          )}
          <UseSkillButton skill={skill} showLabel />
        </div>
      </div>
    </Card>
  );
};
