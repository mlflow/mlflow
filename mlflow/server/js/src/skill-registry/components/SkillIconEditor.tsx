import { PuzzleIcon, Typography, useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';

import { RegistryIconEditor } from '../../common/components/RegistryIconEditor';
import type { RegistryIcon } from '../types';

export const SkillIconEditor = ({
  icons,
  onChange,
}: {
  icons: RegistryIcon[];
  onChange: (icons: RegistryIcon[]) => void;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div css={{ display: 'flex', flexDirection: 'column', gap: theme.spacing.xs }}>
      <Typography.Text bold>
        <FormattedMessage defaultMessage="Icons" description="Label for the skill icon editor" />
      </Typography.Text>
      <RegistryIconEditor
        icons={icons}
        onChange={onChange}
        defaultIcon={<PuzzleIcon aria-hidden />}
        componentId="mlflow.skill_registry.icon_editor"
      />
    </div>
  );
};
