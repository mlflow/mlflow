import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { getArtifactBlob, getArtifactLocationUrl } from '../../../common/utils/ArtifactUtils';
import { ArtifactViewErrorState } from './ArtifactViewErrorState';
import { ArtifactViewSkeleton } from './ArtifactViewSkeleton';

const waveSurferStyling = {
  waveColor: '#1890ff',
  progressColor: '#0b3574',
  height: 500,
};

export type ShowArtifactAudioViewProps = {
  runUuid: string;
  path: string;
  getArtifact?: (...args: any[]) => any;
};

const ShowArtifactAudioView = ({ runUuid, path, getArtifact = getArtifactBlob }: ShowArtifactAudioViewProps) => {
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

    const artifactUrl = getArtifactLocationUrl(path, runUuid);
    getArtifact(artifactUrl)
      .then((blob: Blob) => {
        if (cancelled || !containerRef.current) return;

        blobUrl = URL.createObjectURL(blob);

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
  }, [containerRef, path, runUuid, getArtifact]);

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
