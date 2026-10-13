import { beforeEach, describe, expect, it, jest } from '@jest/globals';

const mockDownloadImage = jest.fn<(figure: unknown, options: unknown) => Promise<void>>();
const mockDisplayGlobalErrorNotification = jest.fn<(message: string) => void>();
let mockPlotlyFactoryLoadCount = 0;

jest.mock('../../PlotlyFactory', () => {
  mockPlotlyFactoryLoadCount += 1;
  return {
    __esModule: true,
    Plotly: {
      downloadImage: mockDownloadImage,
    },
  };
});

jest.mock('../../../../common/utils/Utils', () => ({
  __esModule: true,
  default: {
    displayGlobalErrorNotification: mockDisplayGlobalErrorNotification,
  },
}));

describe('useChartImageDownloadHandler', () => {
  beforeEach(() => {
    jest.resetModules();
    mockDownloadImage.mockReset();
    mockDisplayGlobalErrorNotification.mockReset();
    mockPlotlyFactoryLoadCount = 0;
  });

  it('lazy-loads Plotly only when exporting chart images', async () => {
    mockDownloadImage.mockResolvedValue(undefined);

    const { createChartImageDownloadHandler } = await import('./useChartImageDownloadHandler');

    const handler = createChartImageDownloadHandler(
      [
        {
          type: 'scatter',
          x: [1, 2],
          y: [3, 4],
        },
      ],
      {
        title: 'Accuracy',
      },
    );

    expect(mockPlotlyFactoryLoadCount).toBe(0);

    await handler('png', 'accuracy-chart');

    expect(mockPlotlyFactoryLoadCount).toBe(1);
    expect(mockDownloadImage).toHaveBeenCalledWith(
      {
        data: [
          {
            type: 'scatter',
            x: [1, 2],
            y: [3, 4],
          },
        ],
        layout: {
          title: 'Accuracy',
          paper_bgcolor: 'white',
          plot_bgcolor: 'white',
        },
        config: {
          displaylogo: false,
          modeBarButtonsToRemove: ['toImage'],
        },
      },
      {
        width: 1200,
        height: 600,
        format: 'png',
        filename: 'accuracy-chart',
      },
    );

    await handler('svg', 'accuracy-chart');

    expect(mockPlotlyFactoryLoadCount).toBe(1);
    expect(mockDownloadImage).toHaveBeenCalledTimes(2);
  });

  it('shows a global error notification when exporting chart images fails', async () => {
    mockDownloadImage.mockRejectedValue(new Error('Canvas unavailable'));

    const { createChartImageDownloadHandler } = await import('./useChartImageDownloadHandler');

    const handler = createChartImageDownloadHandler(
      [
        {
          type: 'scatter',
          x: [1, 2],
          y: [3, 4],
        },
      ],
      {
        title: 'Accuracy',
      },
    );

    await expect(handler('png', 'accuracy-chart')).resolves.toBeUndefined();

    expect(mockDisplayGlobalErrorNotification).toHaveBeenCalledWith(
      'Failed to export chart image. Error: Canvas unavailable',
    );
  });
});
