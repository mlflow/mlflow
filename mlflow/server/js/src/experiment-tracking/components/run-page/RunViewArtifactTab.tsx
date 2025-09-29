import { useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity } from '../../../common/types';
import ArtifactPage from '../ArtifactPage';
import { useMediaQuery } from '@databricks/web-shared/hooks';
import type { UseGetRunQueryResponseOutputs } from './hooks/useGetRunQuery';

/**
 * A run page tab containing the artifact browser
 */
export const RunViewArtifactTab = ({
  runTags,
  experimentId,
  runOutputs,
  artifactUri,
  runUuid,
}: {
  runUuid: string;
  experimentId: string;
  artifactUri?: string;
  runOutputs?: UseGetRunQueryResponseOutputs;
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
        runOutputs={runOutputs}
        useAutoHeight={useFullHeightPage}
        artifactRootUri={artifactUri}
        experimentId={experimentId}
      />
    </div>
  );
};
