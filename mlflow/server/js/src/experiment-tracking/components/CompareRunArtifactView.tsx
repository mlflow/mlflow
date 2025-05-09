import { useState } from 'react';
import ShowArtifactPage from './artifact-view-components/ShowArtifactPage';
import { RunInfoEntity } from '../types';
import { useRunsArtifacts } from './experiment-page/hooks/useRunsArtifacts';
import { getCommonArtifacts } from './experiment-page/utils/getCommonArtifacts';
import { useDesignSystemTheme } from '@databricks/design-system';
import { FormattedMessage } from 'react-intl';
import { ArtifactViewTree } from './ArtifactViewTree';
import { getBasename } from '../../common/utils/FileUtils';

export const CompareRunArtifactView = ({
  runUuids,
  runInfos,
  colWidth,
}: {
  runUuids: string[];
  runInfos: RunInfoEntity[];
  colWidth: number;
}) => {
  const { theme } = useDesignSystemTheme();
  const [artifactPath, setArtifactPath] = useState<string | undefined>();

  const { artifactsKeyedByRun } = useRunsArtifacts(runUuids);
  const commonArtifacts = getCommonArtifacts(artifactsKeyedByRun);

  if (commonArtifacts.length === 0) {
    return (
      <h2>
        <FormattedMessage
          defaultMessage="No common artifacts to display."
          description="Text shown when there are no common artifacts between the runs"
        />
      </h2>
    );
  }

  return (
    <div
      css={{
        display: 'flex',
        flexDirection: 'row',
        height: '100vh',
        overflow: 'hidden',
      }}
    >
      <div
        css={{
          backgroundColor: theme.colors.backgroundPrimary,
          color: theme.colors.textPrimary,
          width: '25%',
          whiteSpace: 'nowrap',
          border: `1px solid ${theme.colors.grey300}`,
          overflowY: 'auto',
        }}
      >
        <ArtifactViewTree
          data={commonArtifacts.map((path: string) => ({
            id: path,
            active: artifactPath === path,
            name: getBasename(path),
          }))}
          onToggleTreebeard={({ id }) => setArtifactPath(id)}
        />
      </div>
      <div
        css={{
          border: `1px solid ${theme.colors.grey300}`,
          borderLeft: 'none',
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          overflow: 'hidden',
          height: '100%',
        }}
      >
        <div
          css={{
            display: 'flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
            height: '100%',
            overflow: 'auto',
            gap: '16px',
            padding: '16px',
          }}
        >
          {runUuids.map((runUuid, index) => {
            return (
              <div
                key={runUuid}
                css={{
                  width: `${colWidth}px`,
                  minWidth: '500px',
                  flex: `1 1 ${colWidth}px`,
                  height: artifactPath ? 'calc(100% - 32px)' : 'auto',
                  display: 'flex',
                  flexDirection: 'column',
                  overflow: 'hidden',
                }}
              >
                <div
                  css={{
                    padding: '8px 12px',
                    borderBottom: `1px solid ${theme.colors.grey200}`,
                    fontWeight: 500,
                  }}
                >
                  Run: {runUuid}
                </div>
                <div
                  css={{
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <ShowArtifactPage
                    runUuid={runUuid}
                    artifactRootUri={runInfos[index].artifactUri}
                    path={artifactPath}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
