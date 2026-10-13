import { beforeEach, describe, expect, it, jest } from '@jest/globals';

const mockRegister = jest.fn();
const mockPlotlyCore = { register: mockRegister };
const mockScatterTrace = { name: 'scatter' };
const mockScatterglTrace = { name: 'scattergl' };
const mockBarTrace = { name: 'bar' };
const mockBoxTrace = { name: 'box' };
const mockContourTrace = { name: 'contour' };
const mockParcoordsTrace = { name: 'parcoords' };
const mockPlotlyComponent = { componentName: 'PlotlyFactory' };
const mockCreatePlotlyComponent = jest.fn<(plotly: unknown) => unknown>().mockReturnValue(mockPlotlyComponent);

jest.mock('plotly.js/lib/core', () => ({
  __esModule: true,
  default: mockPlotlyCore,
}));

jest.mock('plotly.js/lib/scatter', () => ({
  __esModule: true,
  default: mockScatterTrace,
}));

jest.mock('plotly.js/lib/scattergl', () => ({
  __esModule: true,
  default: mockScatterglTrace,
}));

jest.mock('plotly.js/lib/bar', () => ({
  __esModule: true,
  default: mockBarTrace,
}));

jest.mock('plotly.js/lib/box', () => ({
  __esModule: true,
  default: mockBoxTrace,
}));

jest.mock('plotly.js/lib/contour', () => ({
  __esModule: true,
  default: mockContourTrace,
}));

jest.mock('plotly.js/lib/parcoords', () => ({
  __esModule: true,
  default: mockParcoordsTrace,
}));

jest.mock('react-plotly.js/factory', () => ({
  __esModule: true,
  default: mockCreatePlotlyComponent,
}));

describe('PlotlyFactory', () => {
  beforeEach(() => {
    jest.resetModules();
    mockRegister.mockReset();
    mockCreatePlotlyComponent.mockReset();
    mockCreatePlotlyComponent.mockReturnValue(mockPlotlyComponent);
  });

  it('registers the supported traces for experiment charts', async () => {
    const plotlyFactoryModule = await import('./PlotlyFactory');

    expect(mockRegister).toHaveBeenCalledWith([
      mockScatterTrace,
      mockScatterglTrace,
      mockBarTrace,
      mockBoxTrace,
      mockContourTrace,
      mockParcoordsTrace,
    ]);
    expect(mockCreatePlotlyComponent).toHaveBeenCalledWith(mockPlotlyCore);
    expect(plotlyFactoryModule.Plotly).toBe(mockPlotlyCore);
    expect(plotlyFactoryModule.default).toBe(mockPlotlyComponent);
  });
});
