import { useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity } from '../../types';
import ArtifactPage from '../ArtifactPage';

/**
 * A run page tab containing the artifact browser
 */
export const RunViewArtifactTab = ({
  runTags,
  runUuid,
}: {
  runUuid: string;
  runTags: Record<string, KeyValueEntity>;
}) => {
  const { theme } = useDesignSystemTheme();
  return (
    <div
      css={{
        flex: 1,
        overflow: 'hidden',
        display: 'flex',
        paddingBottom: theme.spacing.md,
        position: 'relative',
      }}
    >
      <ArtifactPage runUuid={runUuid} runTags={runTags} useAutoHeight />
    </div>
  );
};
