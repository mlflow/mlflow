import { Tag } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

export const SkillRegistryBetaTag = () => (
  <Tag componentId="mlflow.skill_registry.beta_tag" color="turquoise">
    <FormattedMessage defaultMessage="Beta" description="Skill Registry beta feature tag" />
  </Tag>
);
