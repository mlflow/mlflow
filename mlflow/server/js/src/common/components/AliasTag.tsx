import { Tag, useDesignSystemTheme } from '@databricks/design-system';
import type { TagProps } from '@databricks/design-system';

type ModelVersionAliasTagProps = { value: string; compact?: boolean } & Pick<
  TagProps,
  'closable' | 'onClose' | 'className'
>;

// When displayed in compact mode (e.g. within <Select>), constrain the width to 160 pixels
const COMPACT_MODE_MAX_WIDTH = 160;
const REGULAR_MAX_WIDTH = 300;
const TAG_SYMBOL = '@';

export const AliasTag = ({ value, closable, onClose, className, compact = false }: ModelVersionAliasTagProps) => {
  const { theme } = useDesignSystemTheme();
  return (
    <Tag
      componentId="codegen_mlflow_app_src_model-registry_components_aliases_modelversionaliastag.tsx_23"
      css={{
        fontWeight: theme.typography.typographyBoldFontWeight,
        marginRight: theme.spacing.xs,
      }}
      className={className}
      closable={closable}
      onClose={onClose}
      title={value}
    >
      <span
        css={{
          display: 'block',
          whiteSpace: 'nowrap',
          maxWidth: compact ? COMPACT_MODE_MAX_WIDTH : REGULAR_MAX_WIDTH,
          textOverflow: 'ellipsis',
          overflow: 'hidden',
        }}
      >
        {TAG_SYMBOL}&nbsp;{value}
      </span>
    </Tag>
  );
};
