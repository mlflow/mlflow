import { useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity } from '../../types';
import ArtifactPage from '../ArtifactPage';
import { useMediaQuery } from '@databricks/web-shared/hooks';

/**
 * A run page tab containing the artifact browser
 */
export const RunViewArtifactTab = ({
  runTags,
  artifactUri,
  runUuid,
}: {
  runUuid: string;
  experimentId: string;
  artifactUri?: string;
  runTags: Record<string, KeyValueEntity>;
}) => {
  const { theme } = useDesignSystemTheme();

  // Use scrollable artifact area only for non-xs screens
  const useFullHeightPage = useMediaQuery(`(min-width: ${theme.responsive.breakpoints.sm}px)`);

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
      <ArtifactPage
        runUuid={runUuid}
        runTags={runTags}
        useAutoHeight={useFullHeightPage}
        artifactRootUri={artifactUri}
      />
    </div>
  );
};
