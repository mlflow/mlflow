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
  const [waveSurfer, setWaveSurfer] = useState<WaveSurfer | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [audioData, setAudioData] = useState<any>(null);

  useEffect(() => {
    const fetchAudio = async () => {
      setLoading(true);
      try {
        const artifactLocation = getArtifactLocationUrl(path, runUuid);
        const artifactAudioData = await getArtifact(artifactLocation);
        setAudioData({ data: artifactAudioData });
      } catch (error) {
        setError(error as Error);
      } finally {
        setLoading(false);
      }
    };

    fetchAudio();
  }, [runUuid, path, getArtifact]);

  useEffect(() => {
    if (!containerRef.current) return;
    const ws = WaveSurfer.create({
      mediaControls: true,
      container: containerRef.current,
      ...waveSurferStyling,
    });
    setWaveSurfer(ws);

    return () => {
      ws.destroy();
    };
  }, [containerRef]);

  useEffect(() => {
    if (audioData && waveSurfer) {
      waveSurfer.loadBlob(audioData.data);
    }
  }, [audioData, waveSurfer]);

  const showLoading = loading && !error;
  return (
    <>
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
    </>
  );
};

export default ShowArtifactAudioView;
