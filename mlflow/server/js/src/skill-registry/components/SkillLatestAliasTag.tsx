import { Tag, Tooltip } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * Marks the version the reserved latest alias resolves to. The server resolves latest on the fly and never stores
 * it, so callers decide from the skill's latest_version rather than from a version's aliases.
 */
export const SkillLatestAliasTag = () => (
  <Tooltip
    componentId="mlflow.skill_registry.detail.version_latest_alias_tooltip"
    content={
      <FormattedMessage
        defaultMessage="This alias is automatically maintained and always resolves to the newest active version, or to the newest version that is not deleted when none is active."
        description="Tooltip explaining the automatic latest alias on a skill version"
      />
    }
  >
    <span>
      <Tag componentId="mlflow.skill_registry.detail.version_latest_alias" color="brown">
        @ latest
      </Tag>
    </span>
  </Tooltip>
);
