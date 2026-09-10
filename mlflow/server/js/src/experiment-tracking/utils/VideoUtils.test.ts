import { describe, it, expect } from '@jest/globals';
import {
  isVideoArtifactPath,
  extractStepNumber,
  sortVideoArtifacts,
  VIDEO_ARTIFACT_EXTENSIONS,
} from './VideoUtils';
import { AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, TEXT_EXTENSIONS, PDF_EXTENSIONS } from '../../common/utils/FileUtils';

describe('VideoUtils', () => {
  describe('isVideoArtifactPath', () => {
    it('classifies .mp4 as video', () => {
      expect(isVideoArtifactPath('rollout.mp4')).toBe(true);
    });

    it('classifies .MP4 as video (case-insensitive)', () => {
      expect(isVideoArtifactPath('video.MP4')).toBe(true);
    });

    it('classifies .webm as video', () => {
      expect(isVideoArtifactPath('recording.webm')).toBe(true);
    });

    it('classifies .mov as video', () => {
      expect(isVideoArtifactPath('clip.mov')).toBe(true);
    });

    it('classifies .m4v as video', () => {
      expect(isVideoArtifactPath('episode.m4v')).toBe(true);
    });

    it('classifies .ogg as video', () => {
      expect(isVideoArtifactPath('stream.ogg')).toBe(true);
    });

    it('classifies .ogv as video', () => {
      expect(isVideoArtifactPath('clip.ogv')).toBe(true);
    });

    it('handles nested paths', () => {
      expect(isVideoArtifactPath('videos/step_001.mp4')).toBe(true);
      expect(isVideoArtifactPath('media/videos/rollout.webm')).toBe(true);
      expect(isVideoArtifactPath('rollouts/episode_42.mov')).toBe(true);
    });

    it('rejects non-video files', () => {
      expect(isVideoArtifactPath('model.pkl')).toBe(false);
      expect(isVideoArtifactPath('data.csv')).toBe(false);
      expect(isVideoArtifactPath('image.png')).toBe(false);
      expect(isVideoArtifactPath('audio.mp3')).toBe(false);
    });
  });

  describe('mp4 is NOT classified as audio', () => {
    it('.mp4 is not in AUDIO_EXTENSIONS', () => {
      expect(AUDIO_EXTENSIONS.has('mp4')).toBe(false);
    });

    it('.mp4 IS in VIDEO_EXTENSIONS', () => {
      expect(VIDEO_EXTENSIONS.has('mp4')).toBe(true);
    });

    it('.mp4 IS in VIDEO_ARTIFACT_EXTENSIONS', () => {
      expect(VIDEO_ARTIFACT_EXTENSIONS.has('mp4')).toBe(true);
    });
  });

  describe('existing audio formats still work', () => {
    it.each(['m4a', 'mp3', 'wav', 'aac', 'wma', 'flac', 'opus', 'ogg'])(
      '%s is in AUDIO_EXTENSIONS',
      (ext) => {
        expect(AUDIO_EXTENSIONS.has(ext)).toBe(true);
      },
    );
  });

  describe('existing image/text/pdf detection is not broken', () => {
    it.each(['jpg', 'png', 'gif', 'svg', 'bmp', 'jpeg'])('%s is in IMAGE_EXTENSIONS', (ext) => {
      expect(IMAGE_EXTENSIONS.has(ext)).toBe(true);
    });

    it.each(['txt', 'log', 'py', 'json', 'yaml', 'xml'])('%s is in TEXT_EXTENSIONS', (ext) => {
      expect(TEXT_EXTENSIONS.has(ext)).toBe(true);
    });

    it('pdf is in PDF_EXTENSIONS', () => {
      expect(PDF_EXTENSIONS.has('pdf')).toBe(true);
    });
  });

  describe('extractStepNumber', () => {
    it('extracts trailing number after underscore', () => {
      expect(extractStepNumber('step_100.mp4')).toBe(100);
      expect(extractStepNumber('rollout_042.webm')).toBe(42);
    });

    it('extracts trailing number after hyphen', () => {
      expect(extractStepNumber('episode-5.mp4')).toBe(5);
    });

    it('extracts purely numeric filenames', () => {
      expect(extractStepNumber('0003.mp4')).toBe(3);
      expect(extractStepNumber('100.webm')).toBe(100);
    });

    it('returns null for non-numeric filenames', () => {
      expect(extractStepNumber('rollout.mp4')).toBeNull();
      expect(extractStepNumber('my_video.webm')).toBeNull();
    });
  });

  describe('sortVideoArtifacts', () => {
    it('sorts by step number when available', () => {
      const paths = ['videos/step_10.mp4', 'videos/step_2.mp4', 'videos/step_1.mp4'];
      expect(sortVideoArtifacts(paths)).toEqual([
        'videos/step_1.mp4',
        'videos/step_2.mp4',
        'videos/step_10.mp4',
      ]);
    });

    it('sorts alphabetically when no step numbers', () => {
      const paths = ['charlie.mp4', 'alpha.mp4', 'bravo.mp4'];
      expect(sortVideoArtifacts(paths)).toEqual(['alpha.mp4', 'bravo.mp4', 'charlie.mp4']);
    });

    it('puts files with step numbers before files without', () => {
      const paths = ['rollout.mp4', 'step_1.mp4'];
      const sorted = sortVideoArtifacts(paths);
      expect(sorted[0]).toBe('step_1.mp4');
    });

    it('does not mutate the original array', () => {
      const paths = ['b.mp4', 'a.mp4'];
      sortVideoArtifacts(paths);
      expect(paths).toEqual(['b.mp4', 'a.mp4']);
    });
  });
});
