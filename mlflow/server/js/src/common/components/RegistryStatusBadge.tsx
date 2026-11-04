import type { ReactNode } from 'react';
import { Tag, Tooltip, type TagProps } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

/**
 * The header badge of a registry entity whose latest version is not active. Without an active version the entity
 * still resolves to its newest non-deleted version, so that version's status is shown as is; only an entity with
 * nothing to resolve is unavailable.
 */
export const RegistryStatusBadge = ({
  componentId,
  status,
}: {
  /** Prefix for the badge's component IDs, e.g. `mlflow.skill_registry.detail.status`. */
  componentId: string;
  /** The status of the version the entity resolves to, or undefined when it has none. */
  status?: { label: ReactNode; color: TagProps['color']; version: string | number };
}) => (
  <Tooltip
    componentId={`${componentId}.tooltip`}
    content={
      status ? (
        <FormattedMessage
          defaultMessage="No version is active, so this resolves to version {version}. Make a version active to recommend it."
          description="Tooltip for a registry entity whose latest version is not active"
          values={{ version: status.version }}
        />
      ) : (
        <FormattedMessage
          defaultMessage="There is no version to resolve."
          description="Tooltip for a registry entity without any version"
        />
      )
    }
  >
    <span css={{ cursor: 'default' }}>
      <Tag componentId={`${componentId}.tag`} color={status ? status.color : 'coral'}>
        {status ? (
          status.label
        ) : (
          <FormattedMessage
            defaultMessage="Unavailable"
            description="Label for a registry entity without any version"
          />
        )}
      </Tag>
    </span>
  </Tooltip>
);
