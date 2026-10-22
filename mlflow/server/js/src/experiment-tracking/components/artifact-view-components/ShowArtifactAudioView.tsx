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
} & LoggedModelArtifactViewerProps;

const ShowArtifactAudioView = ({
  runUuid,
  path,
  getArtifact = getArtifactBlob,
  artifactRootUri,
  isLoggedModelsMode,
  loggedModelId,
}: ShowArtifactAudioViewProps) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error>();

  useEffect(() => {
    if (!containerRef.current) return;

    setLoading(true);
    setError(undefined);

    let blobUrl: string | undefined;
    let cancelled = false;

    fetchArtifactUnified({ runUuid, path, artifactRootUri, isLoggedModelsMode, loggedModelId }, getArtifact)
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
          setError(new Error('Audio playback failed'));
          ws.destroy();
          wsRef.current = null;
          if (blobUrl) {
            URL.revokeObjectURL(blobUrl);
            blobUrl = undefined;
          }
        });

        wsRef.current = ws;
      })
      .catch((error: Error) => {
        if (!cancelled) {
          setLoading(false);
          setError(error);
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
  }, [containerRef, path, runUuid, getArtifact, artifactRootUri, isLoggedModelsMode, loggedModelId]);

  const showLoading = loading && !error;

  return (
    <div data-testid="audio-artifact-preview">
      {showLoading && <ArtifactViewSkeleton />}
      {error && <ArtifactViewErrorState error={error} />}
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
