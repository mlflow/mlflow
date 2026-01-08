import { LegacySkeleton, useDesignSystemTheme } from '@databricks/design-system';
import type { KeyValueEntity } from '../../../common/types';
import ArtifactPage from '../ArtifactPage';
import { useMediaQuery } from '@databricks/web-shared/hooks';
import { useGetRunQuery, type UseGetRunQueryResponseOutputs } from './hooks/useGetRunQuery';
import { useMemo } from 'react';
import { keyBy } from 'lodash';

/**
 * A run page tab containing the artifact browser
 */
export const RunViewArtifactTab = ({
  runTags: tagsFromSearchRuns,
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

  const runTags = tagsFromSearchRuns;
  const runTagsList = useMemo(() => Object.values(runTags), [runTags]);

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
        entityTags={runTagsList}
      />
    </div>
  );
};
