import Utils from './Utils';
import React from 'react';
import { shallow } from 'enzyme';
import {
  X_AXIS_RELATIVE,
  X_AXIS_STEP,
  X_AXIS_WALL,
} from '../../experiment-tracking/components/MetricsPlotControls';

test('formatMetric', () => {
  expect(Utils.formatMetric(0)).toEqual('0');
  expect(Utils.formatMetric(0.5)).toEqual('0.5');
  expect(Utils.formatMetric(0.001)).toEqual('0.001');

  expect(Utils.formatMetric(0.000123445)).toEqual('1.234e-4');
  expect(Utils.formatMetric(0.000123455)).toEqual('1.235e-4');
  expect(Utils.formatMetric(-0.000123445)).toEqual('-1.234e-4');
  expect(Utils.formatMetric(-0.000123455)).toEqual('-1.235e-4');

  expect(Utils.formatMetric(0.12345)).toEqual('0.123');
  expect(Utils.formatMetric(0.12355)).toEqual('0.124');
  expect(Utils.formatMetric(-0.12345)).toEqual('-0.123');
  expect(Utils.formatMetric(-0.12355)).toEqual('-0.124');

  expect(Utils.formatMetric(1.12345)).toEqual('1.123');
  expect(Utils.formatMetric(1.12355)).toEqual('1.124');
  expect(Utils.formatMetric(-1.12345)).toEqual('-1.123');
  expect(Utils.formatMetric(-1.12355)).toEqual('-1.124');

  expect(Utils.formatMetric(12.12345)).toEqual('12.12');
  expect(Utils.formatMetric(12.12555)).toEqual('12.13');
  expect(Utils.formatMetric(-12.12345)).toEqual('-12.12');
  expect(Utils.formatMetric(-12.12555)).toEqual('-12.13');

  expect(Utils.formatMetric(123.12345)).toEqual('123.1');
  expect(Utils.formatMetric(123.15555)).toEqual('123.2');
  expect(Utils.formatMetric(-123.12345)).toEqual('-123.1');
  expect(Utils.formatMetric(-123.15555)).toEqual('-123.2');

  expect(Utils.formatMetric(1234.12345)).toEqual('1234.1');
  expect(Utils.formatMetric(1234.15555)).toEqual('1234.2');
  expect(Utils.formatMetric(-1234.12345)).toEqual('-1234.1');
  expect(Utils.formatMetric(-1234.15555)).toEqual('-1234.2');

  expect(Utils.formatMetric(1e30)).toEqual('1e+30');
});

test('formatDuration', () => {
  expect(Utils.formatDuration(0)).toEqual('0ms');
  expect(Utils.formatDuration(50)).toEqual('50ms');
  expect(Utils.formatDuration(499)).toEqual('499ms');
  expect(Utils.formatDuration(500)).toEqual('0.5s');
  expect(Utils.formatDuration(900)).toEqual('0.9s');
  expect(Utils.formatDuration(999)).toEqual('1.0s');
  expect(Utils.formatDuration(1000)).toEqual('1.0s');
  expect(Utils.formatDuration(1500)).toEqual('1.5s');
  expect(Utils.formatDuration(2000)).toEqual('2.0s');
  expect(Utils.formatDuration(59 * 1000)).toEqual('59.0s');
  expect(Utils.formatDuration(60 * 1000)).toEqual('1.0min');
  expect(Utils.formatDuration(90 * 1000)).toEqual('1.5min');
  expect(Utils.formatDuration(120 * 1000)).toEqual('2.0min');
  expect(Utils.formatDuration(59 * 60 * 1000)).toEqual('59.0min');
  expect(Utils.formatDuration(60 * 60 * 1000)).toEqual('1.0h');
  expect(Utils.formatDuration(90 * 60 * 1000)).toEqual('1.5h');
  expect(Utils.formatDuration(23 * 60 * 60 * 1000)).toEqual('23.0h');
  expect(Utils.formatDuration(24 * 60 * 60 * 1000)).toEqual('1.0d');
  expect(Utils.formatDuration(36 * 60 * 60 * 1000)).toEqual('1.5d');
  expect(Utils.formatDuration(48 * 60 * 60 * 1000)).toEqual('2.0d');
  expect(Utils.formatDuration(480 * 60 * 60 * 1000)).toEqual('20.0d');
});

test('baseName', () => {
  expect(Utils.baseName('foo')).toEqual('foo');
  expect(Utils.baseName('foo/bar/baz')).toEqual('baz');
  expect(Utils.baseName('/foo/bar/baz')).toEqual('baz');
  expect(Utils.baseName('file:///foo/bar/baz')).toEqual('baz');
});

test('formatSource & renderSource', () => {
  const source_with_name = {
    'mlflow.source.name': { value: 'source' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(source_with_name)).toEqual('source:entry');
  expect(Utils.renderSource(source_with_name)).toEqual('source:entry');

  const source_with_main = {
    'mlflow.source.name': { value: 'source1' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'main' },
  };
  expect(Utils.formatSource(source_with_main)).toEqual('source1');
  expect(Utils.renderSource(source_with_main)).toEqual('source1');

  const source_no_name = {
    'mlflow.source.name': { value: 'source2' },
    'mlflow.source.type': { value: 'PROJECT' },
  };
  expect(Utils.formatSource(source_no_name)).toEqual('source2');
  expect(Utils.renderSource(source_no_name)).toEqual('source2');

  const non_project_source = {
    'mlflow.source.name': { value: 'source3' },
    'mlflow.source.type': { value: 'NOTEBOOK' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(non_project_source)).toEqual('source3');
  expect(Utils.renderSource(non_project_source)).toEqual('source3');

  // formatSource should return a string, renderSource should return an HTML element.
  const github_url = {
    'mlflow.source.name': { value: 'git@github.com:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(github_url)).toEqual('mlflow-apps:entry');
  expect(Utils.renderSource(github_url)).toEqual(
    <a href='https://github.com/mlflow/mlflow-apps' target='_top'>
      mlflow-apps:entry
    </a>,
  );

  const gitlab_url = {
    'mlflow.source.name': { value: 'git@gitlab.com:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(gitlab_url)).toEqual('mlflow-apps:entry');
  expect(Utils.renderSource(gitlab_url)).toEqual(
    <a href='https://gitlab.com/mlflow/mlflow-apps' target='_top'>
      mlflow-apps:entry
    </a>,
  );

  const bitbucket_url = {
    'mlflow.source.name': { value: 'git@bitbucket.org:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(bitbucket_url)).toEqual('mlflow-apps:entry');
  expect(Utils.renderSource(bitbucket_url)).toEqual(
    <a href='https://bitbucket.org/mlflow/mlflow-apps' target='_top'>
      mlflow-apps:entry
    </a>,
  );

  const databricksRunTags = {
    'mlflow.source.name': { value: '/Users/admin/test' },
    'mlflow.source.type': { value: 'NOTEBOOK' },
    'mlflow.databricks.notebookID': { value: '13' },
    'mlflow.databricks.webappURL': { value: 'https://databricks.com' },
  };
  const wrapper = shallow(Utils.renderSource(databricksRunTags));
  expect(wrapper.is('a')).toEqual(true);
  expect(wrapper.props().href).toEqual('http://localhost/#notebook/13');

  const databricksRunRevisionTags = {
    'mlflow.source.name': { value: '/Users/admin/test' },
    'mlflow.source.type': { value: 'NOTEBOOK' },
    'mlflow.databricks.notebookRevisionID': { value: '42' },
    'mlflow.databricks.notebookID': { value: '13' },
    'mlflow.databricks.webappURL': { value: 'https://databricks.com' },
  };
  const wrapper2 = shallow(Utils.renderSource(databricksRunRevisionTags));
  expect(wrapper2.is('a')).toEqual(true);
  expect(wrapper2.props().href).toEqual('http://localhost/#notebook/13/revision/42');

  const wrapper3 = shallow(Utils.renderSource(databricksRunRevisionTags, '?o=123'));
  expect(wrapper3.is('a')).toEqual(true);
  // Query params must appear before the hash, see https://tools.ietf.org/html/rfc3986#section-4.2
  // and https://stackoverflow.com/a/34772568
  expect(wrapper3.props().href).toEqual('http://localhost/?o=123#notebook/13/revision/42');

  const wrapper4 = shallow(Utils.renderSource(databricksRunRevisionTags, '', 'abcd123456'));
  expect(wrapper4.is('a')).toEqual(true);
  expect(wrapper4.props().href).toEqual(
    'http://localhost/#notebook/13/revision/42/mlflow/run/abcd123456',
  );

  const databricksJobTags = {
    'mlflow.source.name': { value: 'job/70/run/5' },
    'mlflow.source.type': { value: 'JOB' },
    'mlflow.databricks.jobID': { value: '70' },
    'mlflow.databricks.jobRunID': { value: '5' },
    'mlflow.databricks.jobType': { value: 'NOTEBOOK' },
    'mlflow.databricks.webappURL': { value: 'https://databricks.com' },
  };
  expect(Utils.formatSource(databricksJobTags)).toEqual('run 5 of job 70');
  const wrapper5 = shallow(Utils.renderSource(databricksJobTags));
  expect(wrapper5.is('a')).toEqual(true);
  expect(wrapper5.props().href).toEqual('http://localhost/#job/70/run/5');
});

test('addQueryParams', () => {
  expect(Utils.setQueryParams('http://localhost/foo', '?o=123')).toEqual(
    'http://localhost/foo?o=123',
  );
  expect(Utils.setQueryParams('http://localhost/foo?param=val', '?o=123')).toEqual(
    'http://localhost/foo?o=123',
  );
  expect(Utils.setQueryParams('http://localhost/foo?param=val', '?param=newval')).toEqual(
    'http://localhost/foo?param=newval',
  );
});

test('dropExtension', () => {
  expect(Utils.dropExtension('foo')).toEqual('foo');
  expect(Utils.dropExtension('foo.xyz')).toEqual('foo');
  expect(Utils.dropExtension('foo.xyz.zyx')).toEqual('foo.xyz');
  expect(Utils.dropExtension('foo/bar/baz.xyz')).toEqual('foo/bar/baz');
  expect(Utils.dropExtension('.foo/.bar/baz.xyz')).toEqual('.foo/.bar/baz');
  expect(Utils.dropExtension('.foo')).toEqual('.foo');
  expect(Utils.dropExtension('.foo.bar')).toEqual('.foo');
  expect(Utils.dropExtension('/.foo')).toEqual('/.foo');
  expect(Utils.dropExtension('.foo/.bar/.xyz')).toEqual('.foo/.bar/.xyz');
});

test('getGitHubRegex', () => {
  const gitHubRegex = Utils.getGitHubRegex();
  const urlAndExpected = [
    [
      'http://github.com/mlflow/mlflow-apps',
      ['/github.com/mlflow/mlflow-apps', 'mlflow', 'mlflow-apps', ''],
    ],
    [
      'https://github.com/mlflow/mlflow-apps',
      ['/github.com/mlflow/mlflow-apps', 'mlflow', 'mlflow-apps', ''],
    ],
    [
      'http://github.com/mlflow/mlflow-apps.git',
      ['/github.com/mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', ''],
    ],
    [
      'https://github.com/mlflow/mlflow-apps.git',
      ['/github.com/mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', ''],
    ],
    [
      'https://github.com/mlflow/mlflow#example/tutorial',
      ['/github.com/mlflow/mlflow#example/tutorial', 'mlflow', 'mlflow', 'example/tutorial'],
    ],
    [
      'https://github.com/username/repo.name#mlproject',
      ['/github.com/username/repo.name#mlproject', 'username', 'repo.name', 'mlproject'],
    ],
    [
      'git@github.com:mlflow/mlflow-apps.git',
      ['@github.com:mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', ''],
    ],
    ['https://some-other-site.com?q=github.com/mlflow/mlflow-apps.git', [null]],
    ['ssh@some-server:mlflow/mlflow-apps.git', [null]],
  ];
  urlAndExpected.forEach((lst) => {
    const url = lst[0];
    const match = url.match(gitHubRegex);
    if (match) {
      match[2] = match[2].replace(/.git/, '');
    }
    expect([].concat(match)).toEqual(lst[1]);
  });
});

test('getMetricPlotStateFromUrl', () => {
  const url0 =
    '?runs=["runUuid1","runUuid2"]&plot_metric_keys=[]' +
    '&plot_layout={"xaxis":{"a": "b"}}&x_axis=step&y_axis_scale=log' +
    '&line_smoothness=0.53&show_point=true&selected_run_ids=["runUuid1"]';
  const url1 =
    '?runs=["runUuid1","runUuid2"]&plot_metric_keys=["metric_1"]&plot_layout={}&x_axis=wall&y_axis_scale=log&show_point=false';
  const url2 = '?runs=["runUuid1","runUuid2"]&plot_metric_keys=["metric_1","metric_2"]';
  // Test extracting plot keys, point info, y axis log scale, line smoothness, layout info
  expect(Utils.getMetricPlotStateFromUrl(url0)).toEqual({
    selectedXAxis: X_AXIS_STEP,
    selectedMetricKeys: [],
    showPoint: true,
    yAxisLogScale: true,
    lineSmoothness: 0.53,
    layout: {
      xaxis: { a: 'b' },
    },
    deselectedCurves: [],
    lastLinearYAxisRange: [],
  });
  expect(Utils.getMetricPlotStateFromUrl(url1)).toEqual({
    selectedXAxis: X_AXIS_WALL,
    selectedMetricKeys: ['metric_1'],
    showPoint: false,
    yAxisLogScale: true,
    lineSmoothness: 0,
    layout: {},
    deselectedCurves: [],
    lastLinearYAxisRange: [],
  });
  expect(Utils.getMetricPlotStateFromUrl(url2)).toEqual({
    selectedXAxis: X_AXIS_RELATIVE,
    selectedMetricKeys: ['metric_1', 'metric_2'],
    showPoint: false,
    yAxisLogScale: false,
    lineSmoothness: 0,
    layout: { autosize: true },
    deselectedCurves: [],
    lastLinearYAxisRange: [],
  });
});

test('getSearchParamsFromUrl', () => {
  const url0 = '?paramKeyFilterString=filt&metricKeyFilterString=metrics&searchInput=';
  const url1 = '?p=&q=&r=';
  const url2 = '?';
  const url3 =
    '?paramKeyFilterString=some=param&metricKeyFilterString=somemetric&searchInput=some-Input';
  expect(Utils.getSearchParamsFromUrl(url0)).toEqual({
    paramKeyFilterString: 'filt',
    metricKeyFilterString: 'metrics',
    searchInput: '',
  });
  expect(Utils.getSearchParamsFromUrl(url1)).toEqual({ p: '', q: '', r: '' });
  expect(Utils.getSearchParamsFromUrl(url2)).toEqual({});
  expect(Utils.getSearchParamsFromUrl(url3)).toEqual({
    paramKeyFilterString: 'some=param',
    metricKeyFilterString: 'somemetric',
    searchInput: 'some-Input',
  });
});

test('getSearchUrlFromState', () => {
  const st0 = {};
  const st1 = { a: 'example' };
  const st2 = { b: 'bbbbbb' };
  const st3 = { param: 'params', metrics: undefined, searchInput: 'someExpression' };
  expect(Utils.getSearchUrlFromState(st0)).toEqual('');
  expect(Utils.getSearchUrlFromState(st1)).toEqual('a=example');
  expect(Utils.getSearchUrlFromState(st2)).toEqual('b=bbbbbb');
  expect(Utils.getSearchUrlFromState(st3)).toEqual(
    'param=params&metrics=&searchInput=someExpression',
  );
});

test('compareExperiments', () => {
  const exp0 = { experiment_id: '0' };
  const exp1 = { experiment_id: '1' };
  const expA = { experiment_id: 'A' };
  const expB = { experiment_id: 'B' };

  expect(Utils.compareExperiments(exp0, exp1)).toEqual(-1);
  expect(Utils.compareExperiments(exp1, exp0)).toEqual(1);
  expect(Utils.compareExperiments(exp1, expA)).toEqual(-1);
  expect(Utils.compareExperiments(expA, expB)).toEqual(-1);

  expect([expB, exp1, expA, exp0].sort(Utils.compareExperiments)).toEqual([exp0, exp1, expA, expB]);
});

test('normalize', () => {
  expect(Utils.normalize('/normalized/absolute/path')).toEqual('/normalized/absolute/path');
  expect(Utils.normalize('normalized/relative/path')).toEqual('normalized/relative/path');
  expect(Utils.normalize('http://mlflow.org/resource')).toEqual('http://mlflow.org/resource');
  expect(Utils.normalize('s3:/bucket/resource')).toEqual('s3:/bucket/resource');
  expect(Utils.normalize('C:\\Windows\\Filesystem\\Path')).toEqual('C:\\Windows\\Filesystem\\Path');

  expect(Utils.normalize('///redundant//absolute/path')).toEqual('/redundant/absolute/path');
  expect(Utils.normalize('redundant//relative///path///')).toEqual('redundant/relative/path');
  expect(Utils.normalize('http://mlflow.org///redundant/')).toEqual('http://mlflow.org/redundant');
  expect(Utils.normalize('s3:///bucket/resource/')).toEqual('s3:/bucket/resource');
});
