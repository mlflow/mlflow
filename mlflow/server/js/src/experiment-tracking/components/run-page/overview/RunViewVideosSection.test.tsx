import { jest, describe, beforeAll, beforeEach, it, expect } from '@jest/globals';
import { render, screen, waitFor } from '../../../../common/utils/TestUtils.react18';
import { IntlProvider } from 'react-intl';
import { DesignSystemProvider } from '@databricks/design-system';
import { RunViewVideosSection } from './RunViewVideosSection';

// Mock the MlflowService
const mockListArtifacts = jest.fn();
jest.mock('../../../sdk/MlflowService', () => ({
  MlflowService: {
    listArtifacts: (...args: any[]) => mockListArtifacts(...args),
  },
}));

// Mock getArtifactBlob to return a dummy blob
jest.mock('../../../../common/utils/ArtifactUtils', () => ({
  ...jest.requireActual<typeof import('../../../../common/utils/ArtifactUtils')>('../../../../common/utils/ArtifactUtils'),
  getArtifactBlob: jest.fn(() => Promise.resolve(new Blob(['video-data'], { type: 'video/mp4' }))),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <IntlProvider locale="en">
    <DesignSystemProvider>{children}</DesignSystemProvider>
  </IntlProvider>
);

describe('RunViewVideosSection', () => {
  beforeAll(() => {
    jest.spyOn(global.URL, 'createObjectURL').mockImplementation(() => 'blob://test-video-url');
    global.URL.revokeObjectURL = jest.fn();
  });

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders a video player with step slider for video artifacts', async () => {
    mockListArtifacts.mockImplementation(({ path }: { run_uuid: string; path?: string }) => {
      if (!path) {
        return Promise.resolve({
          files: [
            { path: 'videos', is_dir: true },
            { path: 'model.pkl', is_dir: false },
          ],
        });
      }
      if (path === 'videos') {
        return Promise.resolve({
          files: [
            { path: 'videos/step_0.mp4', is_dir: false },
            { path: 'videos/step_100.mp4', is_dir: false },
            { path: 'videos/step_500.mp4', is_dir: false },
          ],
        });
      }
      return Promise.resolve({ files: [] });
    });

    render(<RunViewVideosSection runUuid="test-run-123" />, { wrapper });

    await waitFor(() => {
      // Should render a single video player (W&B style: one at a time)
      const video = screen.getByLabelText('video');
      expect(video).toBeInTheDocument();
      expect(video).toHaveAttribute('controls');
    });

    // Should show step label
    expect(screen.getByText(/Step:/)).toBeInTheDocument();
  });

  it('video src points to a blob URL (fetched with auth)', async () => {
    mockListArtifacts.mockResolvedValue({
      files: [{ path: 'videos/rollout.mp4', is_dir: false }],
    });

    render(<RunViewVideosSection runUuid="run-456" />, { wrapper });

    await waitFor(() => {
      const video = screen.getByLabelText('video');
      const src = video.getAttribute('src');
      expect(src).toBe('blob://test-video-url');
    });
  });

  it('does not render non-video artifacts', async () => {
    mockListArtifacts.mockResolvedValue({
      files: [
        { path: 'model.pkl', is_dir: false },
        { path: 'metrics.json', is_dir: false },
        { path: 'image.png', is_dir: false },
        { path: 'clip.mp4', is_dir: false },
      ],
    });

    render(<RunViewVideosSection runUuid="test-run" />, { wrapper });

    await waitFor(() => {
      // Only the mp4 should be rendered as video
      const video = screen.getByLabelText('video');
      expect(video).toBeInTheDocument();
    });
  });

  it('renders nothing when no video artifacts are found', async () => {
    mockListArtifacts.mockResolvedValue({
      files: [
        { path: 'model.pkl', is_dir: false },
        { path: 'data.csv', is_dir: false },
      ],
    });

    render(<RunViewVideosSection runUuid="test-run" />, { wrapper });

    await waitFor(() => {
      expect(screen.queryByLabelText('video')).not.toBeInTheDocument();
    });

    expect(screen.queryByText('Videos')).not.toBeInTheDocument();
  });

  it('renders video with autoPlay for W&B-style experience', async () => {
    mockListArtifacts.mockResolvedValue({
      files: [{ path: 'recording.webm', is_dir: false }],
    });

    render(<RunViewVideosSection runUuid="test-run" />, { wrapper });

    await waitFor(() => {
      const video = screen.getByLabelText('video');
      expect(video).toHaveAttribute('autoplay');
    });
  });

  it('displays the artifact filename below the video', async () => {
    mockListArtifacts.mockResolvedValue({
      files: [{ path: 'videos/my_rollout.mp4', is_dir: false }],
    });

    render(<RunViewVideosSection runUuid="test-run" />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText('my_rollout.mp4')).toBeInTheDocument();
    });
  });

  it('shows video count in header', async () => {
    mockListArtifacts.mockImplementation(({ path }: { run_uuid: string; path?: string }) => {
      if (!path) {
        return Promise.resolve({
          files: [{ path: 'videos', is_dir: true }],
        });
      }
      if (path === 'videos') {
        return Promise.resolve({
          files: [
            { path: 'videos/step_0.mp4', is_dir: false },
            { path: 'videos/step_1.mp4', is_dir: false },
            { path: 'videos/step_2.mp4', is_dir: false },
          ],
        });
      }
      return Promise.resolve({ files: [] });
    });

    render(<RunViewVideosSection runUuid="test-run" />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText('(3)')).toBeInTheDocument();
    });
  });
});
