import Routes from './routes';

jest.mock('../common/utils/RoutingUtils', () => ({
  ...jest.requireActual<typeof import('../common/utils/RoutingUtils')>('../common/utils/RoutingUtils'),
  createMLflowRoutePath: (route: string) => route,
}));

describe('experiment tracking page routes', () => {
  test('yields correct route paths for simple pages', () => {
    expect(Routes.experimentsObservatoryRoute).toEqual('/experiments');
    expect(Routes.rootRoute).toEqual('/');
    expect(Routes.compareRunPageRoute).toEqual('/compare-runs');
    expect(Routes.compareExperimentsPageRoute).toEqual('/compare-experiments');
  });

  test('yields correct route paths for experiment page routes', () => {
    expect(Routes.getExperimentPageRoute('123')).toEqual('/experiments/123');
    expect(Routes.getCompareExperimentsPageRoute(['123', '124'])).toEqual(
      '/compare-experiments/s?experiments=["123","124"]',
    );
    expect(Routes.searchRunsByLifecycleStage('123', 'ACTIVE')).toEqual('/experiments/123?lifecycleFilter=ACTIVE');
    expect(Routes.searchRunsByUser('123', '987654321')).toEqual(
      "/experiments/123?searchFilter=attributes.user_id%20%3D%20'987654321'",
    );
  });

  test('yields correct route paths for run page routes', () => {
    expect(Routes.getRunPageRoute('1234', 'run_uuid_1')).toEqual('/experiments/1234/runs/run_uuid_1');
    expect(Routes.getRunPageRoute('1234', 'run_uuid_1', 'sample/path/to/artifact')).toEqual(
      '/experiments/1234/runs/run_uuid_1/artifacts/sample/path/to/artifact',
    );
    expect(Routes.getCompareRunPageRoute(['run_uuid_1', 'run_uuid_2'], ['123', '124'])).toEqual(
      '/compare-runs?runs=["run_uuid_1","run_uuid_2"]&experiments=["123","124"]',
    );
  });

  test('yields correct route paths for metric page route', () => {
    expect(
      Routes.getMetricPageRoute(
        // Run UUIDs
        ['run_uuid_1', 'run_uuid_2'],
        // Main metric key
        'primary_metric_key',
        // Experiment IDs
        ['123', '124'],
        // Plot metric keys
        ['metric_key_1', 'metric_key_2'],
        // Mocked plot layout
        { some_plot_layout: 'layout_value' },
        // Selected X Axis
        'relative',
        // Logarithmic Y axis
        true,
        // Line smoothness
        2,
        // Showing point
        false,
        // DeselectedCurves
        [],
        // Last X range
        [],
      ),
    ).toEqual(
      '/metric?runs=["run_uuid_1","run_uuid_2"]&metric=%22primary_metric_key%22&experiments=["123","124"]&plot_metric_keys=%5B%22metric_key_1%22%2C%22metric_key_2%22%5D&plot_layout={"some_plot_layout":"layout_value"}&x_axis=relative&y_axis_scale=log&line_smoothness=2&show_point=false&deselected_curves=[]&last_linear_y_axis_range=[]',
    );
  });

  test('yields correct route paths for page route of a metric with special chars', () => {
    const route = Routes.getMetricPageRoute(
      // Run UUIDs
      ['run_uuid_1', 'run_uuid_2'],
      // Main metric key
      'some metric with !#@$~*& special chars',
      // Experiment IDs
      ['123', '124'],
      // Plot metric keys
      ['another #@! weird metric', 'metric_key_2'],
      // Mocked plot layout
      { some_plot_layout: 'layout_value' },
      // Selected X Axis
      'relative',
      // Logarithmic Y axis
      true,
      // Line smoothness
      2,
      // Showing point
      false,
      // DeselectedCurves
      [],
      // Last X range
      [],
    );

    const parsedUrl = Object.fromEntries(new URLSearchParams(route).entries());

    const resultMetricKey = JSON.parse(parsedUrl['metric']);
    const resultPlotMetricKeys = JSON.parse(parsedUrl['plot_metric_keys']);

    expect(resultMetricKey).toEqual('some metric with !#@$~*& special chars');
    expect(resultPlotMetricKeys).toEqual(['another #@! weird metric', 'metric_key_2']);
  });
});
