import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { getArtifactBlob } from '../../../common/utils/ArtifactUtils';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';
import type { LoggedModelArtifactViewerProps } from './ArtifactViewComponents.types';
import { fetchArtifactUnified } from './utils/fetchArtifactUnified';

const waveSurferStyling = {
  waveColor: '#1890ff',
  progressColor: '#0b3574',
  height: 500,
};

export type ShowArtifactAudioViewProps = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
} & Omit<LoggedModelArtifactViewerProps, 'experimentId'> & { experimentId?: string };

const ShowArtifactAudioView = ({
  runUuid,
  path,
  getArtifact = getArtifactBlob,
  artifactRootUri,
  isLoggedModelsMode,
  loggedModelId,
  experimentId,
  entityTags,
}: ShowArtifactAudioViewProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<boolean>(false);

  useEffect(() => {
    if (!containerRef.current) return;

    setLoading(true);
    setError(false);

    let blobUrl: string | undefined;
    let cancelled = false;

    fetchArtifactUnified(
      { runUuid, path, artifactRootUri, isLoggedModelsMode, loggedModelId, experimentId, entityTags },
      getArtifact,
    )
      .then((result) => {
        if (cancelled || !containerRef.current) return;

        blobUrl = URL.createObjectURL(result as Blob);

        const ws = WaveSurfer.create({
          mediaControls: true,
          container: containerRef.current,
          ...waveSurferStyling,
          url: blobUrl,
        });

        ws.on('ready', () => {
          setLoading(false);
        });

        ws.on('error', () => {
          setLoading(false);
          setError(true);
          ws.destroy();
          wsRef.current = null;
          if (blobUrl) {
            URL.revokeObjectURL(blobUrl);
            blobUrl = undefined;
          }
        });

        wsRef.current = ws;
      })
      .catch(() => {
        if (!cancelled) {
          setLoading(false);
          setError(true);
        }
      });

    return () => {
      cancelled = true;
      if (wsRef.current) {
        wsRef.current.destroy();
        wsRef.current = null;
      }
      if (blobUrl) {
        URL.revokeObjectURL(blobUrl);
      }
    };
  }, [
    containerRef,
    path,
    runUuid,
    getArtifact,
    artifactRootUri,
    isLoggedModelsMode,
    loggedModelId,
    experimentId,
    entityTags,
  ]);

  const showLoading = loading && !error;

  return (
    <div data-testid="audio-artifact-preview">
      {showLoading && <ArtifactViewSkeleton />}
      {error && <ArtifactViewErrorState />}
      {/* This div is always rendered, but its visibility is controlled by the loading and error states */}
      <div
        css={{
          display: loading || error ? 'none' : 'block',
          padding: 20,
        }}
      >
        <div ref={containerRef} />
      </div>
    </div>
  );
};

export default ShowArtifactAudioView;
