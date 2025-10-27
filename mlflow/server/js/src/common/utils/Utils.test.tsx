/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import Utils from './Utils';
import React from 'react';
import { X_AXIS_RELATIVE, X_AXIS_STEP, X_AXIS_WALL } from '../../experiment-tracking/components/MetricsPlotControls';
import { RunTag } from '../../experiment-tracking/sdk/MlflowMessages';

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

test('getDuration', () => {
  expect(Utils.getDuration(1, null)).toEqual(null);
  expect(Utils.getDuration(1, undefined)).toEqual(null);
  expect(Utils.getDuration(null, 1)).toEqual(null);
  expect(Utils.getDuration(undefined, 1)).toEqual(null);
  expect(Utils.getDuration(undefined, undefined)).toEqual(null);
  expect(Utils.getDuration(null, null)).toEqual(null);
  expect(Utils.getDuration(1, 11)).toEqual('10ms');
  expect(Utils.getDuration(1, 501)).toEqual('0.5s');
  expect(Utils.getDuration(1, 901)).toEqual('0.9s');
  expect(Utils.getDuration(1, 60001)).toEqual('1.0min');
  expect(Utils.getDuration(1, 60 * 60 * 1000 + 1)).toEqual('1.0h');
  expect(Utils.getDuration(1, 24 * 60 * 60 * 1000 + 1)).toEqual('1.0d');
});

test('baseName', () => {
  expect(Utils.baseName('foo')).toEqual('foo');
  expect(Utils.baseName('foo/bar/baz')).toEqual('baz');
  expect(Utils.baseName('/foo/bar/baz')).toEqual('baz');
  expect(Utils.baseName('file:///foo/bar/baz')).toEqual('baz');
});

test('renderNotebookSource', () => {
  const notebookId = '12345678';
  const revisionId = '987654';
  const runUuid = '1133557799';
  const sourceName = '/Users/test/iris_feature';
  const nameOverride = 'some feature';
  const queryParams = '?o=123456789';

  expect(Utils.renderNotebookSource(null, null, null, null, sourceName, null)).toEqual('iris_feature');
  expect(Utils.renderNotebookSource(null, notebookId, null, null, sourceName, null)).toEqual(
    <a title={sourceName} href={`http://localhost/#notebook/${notebookId}`} target="_top">
      iris_feature
    </a>,
  );
  expect(Utils.renderNotebookSource(null, notebookId, revisionId, null, sourceName, null)).toEqual(
    <a title={sourceName} href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}`} target="_top">
      iris_feature
    </a>,
  );
  expect(Utils.renderNotebookSource(null, notebookId, revisionId, runUuid, sourceName, null)).toEqual(
    <a
      title={sourceName}
      href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      iris_feature
    </a>,
  );
  expect(Utils.renderNotebookSource(null, notebookId, revisionId, runUuid, null, null)).toEqual(
    <a
      title={Utils.getDefaultNotebookRevisionName(notebookId, revisionId)}
      href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      {Utils.getDefaultNotebookRevisionName(notebookId, revisionId)}
    </a>,
  );
  expect(Utils.renderNotebookSource(null, notebookId, revisionId, runUuid, sourceName, null, nameOverride)).toEqual(
    <a
      title={sourceName}
      href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      {nameOverride}
    </a>,
  );
  expect(Utils.renderNotebookSource(queryParams, notebookId, revisionId, runUuid, sourceName, null)).toEqual(
    <a
      title={sourceName}
      href={`http://localhost/${queryParams}#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      iris_feature
    </a>,
  );
  expect(
    Utils.renderNotebookSource(
      queryParams,
      notebookId,
      revisionId,
      runUuid,
      sourceName,
      // @ts-expect-error TS(2345): Argument of type '"http://databricks"' is not assi... Remove this comment to see the full error message
      'http://databricks',
      null,
    ),
  ).toEqual(
    <a
      title={sourceName}
      href={`http://databricks/${queryParams}#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      iris_feature
    </a>,
  );
});

test('renderJobSource', () => {
  const jobId = '123456';
  const jobRunId = '98765';
  const jobName = 'job xxx';
  const nameOverride = 'random text';
  const queryParams = '?o=123456789';

  expect(Utils.renderJobSource(null, null, null, jobName, null)).toEqual(jobName);
  expect(Utils.renderJobSource(null, jobId, null, jobName, null)).toEqual(
    <a title={jobName} href={`http://localhost/#job/${jobId}`} target="_top">
      {jobName}
    </a>,
  );
  expect(Utils.renderJobSource(null, jobId, null, null, null)).toEqual(
    <a title={`job ${jobId}`} href={`http://localhost/#job/${jobId}`} target="_top">
      {`job ${jobId}`}
    </a>,
  );
  expect(Utils.renderJobSource(null, jobId, jobRunId, jobName, null)).toEqual(
    <a title={jobName} href={`http://localhost/#job/${jobId}/run/${jobRunId}`} target="_top">
      {jobName}
    </a>,
  );
  expect(Utils.renderJobSource(null, jobId, jobRunId, null, null)).toEqual(
    <a
      title={Utils.getDefaultJobRunName(jobId, jobRunId)}
      href={`http://localhost/#job/${jobId}/run/${jobRunId}`}
      target="_top"
    >
      {Utils.getDefaultJobRunName(jobId, jobRunId)}
    </a>,
  );
  expect(Utils.renderJobSource(null, jobId, jobRunId, jobName, null, nameOverride)).toEqual(
    <a title={jobName} href={`http://localhost/#job/${jobId}/run/${jobRunId}`} target="_top">
      {nameOverride}
    </a>,
  );
  expect(Utils.renderJobSource(queryParams, jobId, jobRunId, jobName, null)).toEqual(
    <a title={jobName} href={`http://localhost/${queryParams}#job/${jobId}/run/${jobRunId}`} target="_top">
      {jobName}
    </a>,
  );
  expect(
    // @ts-expect-error TS(2345): Argument of type '"https://databricks"' is not ass... Remove this comment to see the full error message
    Utils.renderJobSource(queryParams, jobId, jobRunId, jobName, 'https://databricks', null),
  ).toEqual(
    <a title={jobName} href={`https://databricks/${queryParams}#job/${jobId}/run/${jobRunId}`} target="_top">
      {jobName}
    </a>,
  );
});

test('formatSource & renderSource', () => {
  const source_with_name = {
    'mlflow.source.name': { value: 'source' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(source_with_name)).toEqual('source:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(source_with_name)).toEqual('source:entry');

  const source_with_main = {
    'mlflow.source.name': { value: 'source1' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'main' },
  };
  expect(Utils.formatSource(source_with_main)).toEqual('source1');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(source_with_main)).toEqual('source1');

  const source_no_name = {
    'mlflow.source.name': { value: 'source2' },
    'mlflow.source.type': { value: 'PROJECT' },
  };
  expect(Utils.formatSource(source_no_name)).toEqual('source2');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(source_no_name)).toEqual('source2');

  const non_project_source = {
    'mlflow.source.name': { value: 'source3' },
    'mlflow.source.type': { value: 'NOTEBOOK' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(non_project_source)).toEqual('source3');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(non_project_source)).toEqual('source3');

  // formatSource should return a string, renderSource should return an HTML element.
  const github_url = {
    'mlflow.source.name': { value: 'git@github.com:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(github_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(github_url)).toEqual(
    <a href="https://github.com/mlflow/mlflow-apps" target="_top">
      mlflow-apps:entry
    </a>,
  );

  const gitlab_url = {
    'mlflow.source.name': { value: 'git@gitlab.com:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(gitlab_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(gitlab_url)).toEqual(
    <a href="https://gitlab.com/mlflow/mlflow-apps" target="_top">
      mlflow-apps:entry
    </a>,
  );

  const gitlab_long_url = {
    'mlflow.source.name': { value: 'git@gitlab.com:mlflow/mlflow-apps.git#tmp' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(gitlab_long_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(gitlab_long_url)).toEqual(
    <a href="https://gitlab.com/mlflow/mlflow-apps/-/tree/master/tmp" target="_top">
      mlflow-apps:entry
    </a>,
  );

  const bitbucket_url = {
    'mlflow.source.name': { value: 'git@bitbucket.org:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(Utils.formatSource(bitbucket_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(Utils.renderSource(bitbucket_url)).toEqual(
    <a href="https://bitbucket.org/mlflow/mlflow-apps" target="_top">
      mlflow-apps:entry
    </a>,
  );
});

test('setQueryParams', () => {
  expect(Utils.setQueryParams('http://localhost/foo', '?o=123')).toEqual('http://localhost/foo?o=123');
  expect(Utils.setQueryParams('http://localhost/foo?param=val', '?o=123')).toEqual('http://localhost/foo?o=123');
  expect(Utils.setQueryParams('http://localhost/foo?param=val', '?param=newval')).toEqual(
    'http://localhost/foo?param=newval',
  );
  expect(Utils.setQueryParams('https://localhost/foo?param=val', '?param=newval')).toEqual(
    'https://localhost/foo?param=newval',
  );
  expect(Utils.setQueryParams('localhost/foo?param=val', '?param=newval')).toEqual(
    'https://localhost/foo?param=newval',
  );
});

test('ensureUrlScheme', () => {
  expect(Utils.ensureUrlScheme('http://localhost/xyz/abc?o=123')).toEqual('http://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('https://localhost/xyz/abc?o=123')).toEqual('https://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('HTTPS://localhost/xyz/abc?o=123')).toEqual('HTTPS://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('localhost/xyz/abc?o=123')).toEqual('https://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('localhost/xyz/abc?o=123', 'http')).toEqual('http://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('user:pass@localhost/xyz/abc?o=123')).toEqual(
    'https://user:pass@localhost/xyz/abc?o=123',
  );
  expect(Utils.ensureUrlScheme('https://user:pass@localhost/xyz/abc?o=123')).toEqual(
    'https://user:pass@localhost/xyz/abc?o=123',
  );
  expect(Utils.ensureUrlScheme('https://localhost/xyz/abc?o=123', 'http')).toEqual('https://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('://localhost/xyz/abc?o=123', 'https')).toEqual('https://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('://localhost/xyz/abc?o=123', 'ws')).toEqual('ws://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('wss://localhost/xyz/abc?o=123')).toEqual('wss://localhost/xyz/abc?o=123');
  expect(Utils.ensureUrlScheme('scheme-with+symbols.123x://localhost/xyz/abc?o=123')).toEqual(
    'scheme-with+symbols.123x://localhost/xyz/abc?o=123',
  );
  expect(Utils.ensureUrlScheme('legal-schema://abc')).toEqual('legal-schema://abc');
  expect(Utils.ensureUrlScheme('illegal_schema://abc')).toEqual('https://illegal_schema://abc');
  expect(Utils.ensureUrlScheme(undefined)).toEqual(undefined);
});

test('addQueryParams', () => {
  expect(Utils.addQueryParams('', { o: null })).toEqual('');
  expect(Utils.addQueryParams('?param=val', { o: null })).toEqual('?param=val');
  expect(Utils.addQueryParams('', { o: 123 })).toEqual('?o=123');
  expect(Utils.addQueryParams('', { o: 123, param: 'val' })).toEqual('?o=123&param=val');
  expect(Utils.addQueryParams('?param=val', { o: 123 })).toEqual('?param=val&o=123');
  expect(Utils.addQueryParams('?o=456', { o: 123 })).toEqual('?o=123');
});

test('getDefaultJobRunName', () => {
  expect(Utils.getDefaultJobRunName(null, null)).toEqual('-');
  expect(Utils.getDefaultJobRunName(123, null)).toEqual('job 123');
  expect(Utils.getDefaultJobRunName(123, 456)).toEqual('run 456 of job 123');
  // @ts-expect-error TS(2345): Argument of type '7890' is not assignable to param... Remove this comment to see the full error message
  expect(Utils.getDefaultJobRunName(123, 456, 7890)).toEqual('workspace 7890: run 456 of job 123');
});

test('getDefaultNotebookRevisionName', () => {
  expect(Utils.getDefaultNotebookRevisionName(null, null)).toEqual('-');
  expect(Utils.getDefaultNotebookRevisionName(123, null)).toEqual('notebook 123');
  expect(Utils.getDefaultNotebookRevisionName(123, 456)).toEqual('revision 456 of notebook 123');
  // @ts-expect-error TS(2345): Argument of type '7890' is not assignable to param... Remove this comment to see the full error message
  expect(Utils.getDefaultNotebookRevisionName(123, 456, 7890)).toEqual('workspace 7890: revision 456 of notebook 123');
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
    ['http://github.com/mlflow/mlflow-apps', ['/github.com/mlflow/mlflow-apps', 'mlflow', 'mlflow-apps', '']],
    ['https://github.com/mlflow/mlflow-apps', ['/github.com/mlflow/mlflow-apps', 'mlflow', 'mlflow-apps', '']],
    ['http://github.com/mlflow/mlflow-apps.git', ['/github.com/mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', '']],
    ['https://github.com/mlflow/mlflow-apps.git', ['/github.com/mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', '']],
    [
      'https://github.com/mlflow/mlflow#example/tutorial',
      ['/github.com/mlflow/mlflow#example/tutorial', 'mlflow', 'mlflow', 'example/tutorial'],
    ],
    [
      'https://github.com/username/repo.name#mlproject',
      ['/github.com/username/repo.name#mlproject', 'username', 'repo.name', 'mlproject'],
    ],
    ['git@github.com:mlflow/mlflow-apps.git', ['@github.com:mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', '']],
    ['https://some-other-site.com?q=github.com/mlflow/mlflow-apps.git', [null]],
    ['ssh@some-server:mlflow/mlflow-apps.git', [null]],
  ];
  urlAndExpected.forEach((lst) => {
    const url = lst[0];
    const match = (url as any).match(gitHubRegex);
    if (match) {
      match[2] = match[2].replace(/.git/, '');
    }
    expect([].concat(match)).toEqual(lst[1]);
  });
});

test('getGitLabRegex', () => {
  const gitLabRegex = Utils.getGitLabRegex();
  const urlAndExpected = [
    ['http://gitlab.com/mlflow/mlflow-apps', ['/gitlab.com/mlflow/mlflow-apps', 'mlflow', 'mlflow-apps', '']],
    ['https://gitlab.com/mlflow/mlflow-apps', ['/gitlab.com/mlflow/mlflow-apps', 'mlflow', 'mlflow-apps', '']],
    ['http://gitlab.com/mlflow/mlflow-apps.git', ['/gitlab.com/mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', '']],
    ['https://gitlab.com/mlflow/mlflow-apps.git', ['/gitlab.com/mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', '']],
    [
      'https://gitlab.com/mlflow/mlflow#example/tutorial',
      ['/gitlab.com/mlflow/mlflow#example/tutorial', 'mlflow', 'mlflow', 'example/tutorial'],
    ],
    [
      'https://gitlab.com/username/repo.name#mlproject',
      ['/gitlab.com/username/repo.name#mlproject', 'username', 'repo.name', 'mlproject'],
    ],
    ['git@gitlab.com:mlflow/mlflow-apps.git', ['@gitlab.com:mlflow/mlflow-apps.git', 'mlflow', 'mlflow-apps', '']],
    ['https://some-other-site.com?q=gitlab.com/mlflow/mlflow-apps.git', [null]],
    ['ssh@some-server:mlflow/mlflow-apps.git', [null]],
  ];
  urlAndExpected.forEach((lst) => {
    const url = lst[0];
    const match = (url as any).match(gitLabRegex);
    if (match) {
      match[2] = match[2].replace(/.git/, '');
    }
    expect([].concat(match)).toEqual(lst[1]);
  });
});

test('getRegex', () => {
  const gitRegex = Utils.getGitRegex();
  const urlAndExpected = [
    [
      'https://custom.git.domain/repo/directory#project/directory',
      ['https://custom.git.domain', 'repo/directory', 'project/directory'],
    ],
    [
      'git@git.custom.in/repo/directory#project/directory',
      ['git@git.custom.in', 'repo/directory', 'project/directory'],
    ],
    ['https://some-other-site.com?q=github.com/mlflow/mlflow-apps.git', [undefined]],
    ['ssh@some-server:mlflow/mlflow-apps.git', [undefined]],
  ];
  urlAndExpected.forEach((lst) => {
    const url = lst[0];
    const match = (url as any).match(gitRegex);
    if (match) {
      match[2] = match[2].replace(/.git/, '');
    }
    expect([].concat(match?.slice(1))).toEqual(lst[1]);
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
  const url3 =
    '?runs=["runUuid1","runUuid2"]&plot_metric_keys=%5B%22some%20%23%40!%20unusual%20metric%20name%22%2C%22metric_key_2%22%5D';
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
  expect(Utils.getMetricPlotStateFromUrl(url3)).toEqual({
    selectedXAxis: X_AXIS_RELATIVE,
    selectedMetricKeys: ['some #@! unusual metric name', 'metric_key_2'],
    showPoint: false,
    yAxisLogScale: false,
    lineSmoothness: 0,
    layout: { autosize: true },
    deselectedCurves: [],
    lastLinearYAxisRange: [],
  });
});

test('getSearchParamsFromUrl', () => {
  const url0 = '?searchInput=';
  const url1 = '?p=&q=&r=';
  const url2 = '?';
  const url3 = '?searchInput=some-Input';
  const url4 = '?boolVal1=true&boolVal2=false';
  // old style for arrays
  const url5 =
    'categorizedUncheckedKeys%5Bmetrics%5D%5B0%5D=1&categorizedUncheckedKeys%5Bmetrics%5D%5B1%5D=%F0%9F%99%82';
  const url6 = 'categorizedUncheckedKeys[metrics]=1,%F0%9F%99%82'; // new style for arrays
  const url7 = 'a[b][]=c'; // single item array
  const url8 = 'a[]='; // empty array
  const bigArray = [...Array(501).keys()].map((x) => x.toString());
  const bigArrayParams = 'arr=' + bigArray.join(',');
  const bigArrayParamsOldStyle = bigArray.map((i) => 'arr%5B' + i + '%5D=' + i).join('&');
  const tooBigArrayParams = [...Array(502).keys()].map((i) => 'arr%5B' + i + '%5D=' + i).join('&');
  const uncheckedKeysObj = { categorizedUncheckedKeys: { metrics: ['1', '🙂'] } };
  expect(Utils.getSearchParamsFromUrl(url0)).toEqual({
    searchInput: '',
  });
  expect(Utils.getSearchParamsFromUrl(url1)).toEqual({ p: '', q: '', r: '' });
  expect(Utils.getSearchParamsFromUrl(url2)).toEqual({});
  expect(Utils.getSearchParamsFromUrl(url3)).toEqual({
    searchInput: 'some-Input',
  });
  expect(Utils.getSearchParamsFromUrl(url4)).toEqual({
    boolVal1: true,
    boolVal2: false,
  });
  expect(Utils.getSearchParamsFromUrl(url5)).toEqual(uncheckedKeysObj);
  expect(Utils.getSearchParamsFromUrl(url6)).toEqual(uncheckedKeysObj);
  expect(Utils.getSearchParamsFromUrl(url7)).toEqual({ a: { b: ['c'] } });
  expect(Utils.getSearchParamsFromUrl(url8)).toEqual({ a: [''] });
  expect(Utils.getSearchParamsFromUrl(bigArrayParams)).toEqual({ arr: bigArray });
  expect(Utils.getSearchParamsFromUrl(bigArrayParamsOldStyle)).toEqual({ arr: bigArray });
  // @ts-expect-error TS(4111): Property 'arr' comes from an index signature, so i... Remove this comment to see the full error message
  expect(Array.isArray(Utils.getSearchParamsFromUrl(tooBigArrayParams).arr)).toBe(false);
});

test('getSearchUrlFromState', () => {
  const st0 = {};
  const st1 = { a: 'example' };
  const st2 = { b: 'bbbbbb' };
  const st3 = { param: 'params', metrics: undefined, searchInput: 'someExpression' };
  const st4 = { categorizedUncheckedKeys: { metrics: ['1', '2'] } };
  const st5 = { a: [undefined] }; // array with undefined item
  const st6 = { a: ['b'] }; // array with one item
  const st7 = { a: ['🙂'] }; // array with emoji
  expect(Utils.getSearchUrlFromState(st0)).toEqual('');
  expect(Utils.getSearchUrlFromState(st1)).toEqual('a=example');
  expect(Utils.getSearchUrlFromState(st2)).toEqual('b=bbbbbb');
  expect(Utils.getSearchUrlFromState(st3)).toEqual('param=params&metrics=&searchInput=someExpression');
  expect(Utils.getSearchUrlFromState(st4)).toEqual('categorizedUncheckedKeys[metrics]=1,2');
  expect(Utils.getSearchUrlFromState(st5)).toEqual('a[]=');
  expect(Utils.getSearchUrlFromState(st6)).toEqual('a[]=b');
  expect(Utils.getSearchUrlFromState(st7)).toEqual('a[]=%F0%9F%99%82');
});

test('compareExperiments', () => {
  const exp0 = { experimentId: '0' };
  const exp1 = { experimentId: '1' };
  const expA = { experimentId: 'A' };
  const expB = { experimentId: 'B' };

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
test('getLoggedModelsFromTags correctly parses run tag for logged models', () => {
  const tags = {
    'mlflow.log-model.history': (RunTag as any).fromJs({
      key: 'mlflow.log-model.history',
      value: JSON.stringify([
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-31',
          flavors: { keras: {}, python_function: {} },
        },
      ]),
    }),
  };
  const parsed = Utils.getLoggedModelsFromTags(tags);
  expect(parsed).toHaveLength(1);
  expect((parsed as any)[0].artifactPath).toEqual('somePath');
  expect((parsed as any)[0].flavors).toHaveLength(1);
  expect((parsed as any)[0].flavors[0]).toEqual('keras');
});

test('getLoggedModelsFromTags should correctly dedup and sort logged models', () => {
  const tags = {
    'mlflow.log-model.history': (RunTag as any).fromJs({
      key: 'mlflow.log-model.history',
      value: JSON.stringify([
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-29',
          flavors: { keras: {}, python_function: {} },
        },
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-30',
          flavors: { sklearn: {}, python_function: {} },
        },
        {
          run_id: 'run-uuid',
          artifact_path: 'someOtherPath',
          utc_time_created: '2020-10-31',
          flavors: { python_function: {} },
        },
      ]),
    }),
  };
  const filtered = Utils.getLoggedModelsFromTags(tags);
  expect(filtered.length).toEqual(2);
  expect(filtered).toEqual([
    {
      artifactPath: 'someOtherPath',
      flavors: ['pyfunc'],
      utcTimeCreated: 1604102400,
    },
    {
      artifactPath: 'somePath',
      flavors: ['sklearn'],
      utcTimeCreated: 1604016000,
    },
  ]);
});

test('getLoggedModelsFromTags should not crash on invalid JSON', () => {
  const tagValue = JSON.stringify([
    {
      run_id: 'run-uuid',
      artifact_path: 'somePath',
      utc_time_created: '2020-10-29',
      flavors: { keras: {}, python_function: {} },
    },
    {
      run_id: 'run-uuid',
      artifact_path: 'somePath',
      utc_time_created: '2020-10-30',
      flavors: { sklearn: {}, python_function: {} },
    },
    {
      run_id: 'run-uuid',
      artifact_path: 'someOtherPath',
      utc_time_created: '2020-10-31',
      flavors: { python_function: {} },
    },
  ]);

  const tags = {
    'mlflow.log-model.history': {
      key: 'mlflow.log-model.history',
      value: tagValue.slice(10), // truncate the JSON string to make it invalid
    },
  };

  // it should just return an empty array
  const models = Utils.getLoggedModelsFromTags(tags);
  expect(models.length).toEqual(0);
});

test('mergeLoggedAndRegisteredModels should merge logged and registered model', () => {
  const tags = {
    'mlflow.log-model.history': (RunTag as any).fromJs({
      key: 'mlflow.log-model.history',
      value: JSON.stringify([
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-31',
          flavors: { keras: {}, python_function: {} },
        },
      ]),
    }),
  };
  const modelVersions = [
    {
      name: 'someModel',
      version: '3',
      source: 'nananaBatman/artifacts/somePath',
      creation_timestamp: 123456,
      run_id: 'run-uuid',
    },
  ];
  const loggedModels = Utils.getLoggedModelsFromTags(tags);
  const models = Utils.mergeLoggedAndRegisteredModels(loggedModels, modelVersions);
  expect(models).toEqual([
    {
      artifactPath: 'somePath',
      flavors: ['keras'],
      utcTimeCreated: 1604102400,
      registeredModelName: 'someModel',
      registeredModelVersion: '3',
      registeredModelCreationTimestamp: 123456,
    },
  ]);
});

test('mergeLoggedAndRegisteredModels should output 2 logged and 1 registered model', () => {
  const tags = {
    'mlflow.log-model.history': (RunTag as any).fromJs({
      key: 'mlflow.log-model.history',
      value: JSON.stringify([
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-31',
          flavors: { keras: {}, python_function: {} },
        },
        {
          run_id: 'run-uuid',
          artifact_path: 'someOtherPath',
          utc_time_created: '2020-10-31',
          flavors: { sklearn: {}, python_function: {} },
        },
      ]),
    }),
  };
  const modelVersions = [
    {
      name: 'someModel',
      version: '3',
      source: 'nananaBatman/artifacts/somePath',
      run_id: 'run-uuid',
      creation_timestamp: 123456,
    },
  ];
  const loggedModels = Utils.getLoggedModelsFromTags(tags);
  const models = Utils.mergeLoggedAndRegisteredModels(loggedModels, modelVersions);
  expect(models).toEqual([
    {
      artifactPath: 'somePath',
      flavors: ['keras'],
      utcTimeCreated: 1604102400,
      registeredModelName: 'someModel',
      registeredModelVersion: '3',
      registeredModelCreationTimestamp: 123456,
    },
    {
      artifactPath: 'someOtherPath',
      utcTimeCreated: 1604102400,
      flavors: ['sklearn'],
    },
  ]);
});

test('mergeLoggedAndRegisteredModels should output registered models in order', () => {
  const tags = {
    'mlflow.log-model.history': (RunTag as any).fromJs({
      key: 'mlflow.log-model.history',
      value: JSON.stringify([
        {
          run_id: 'run-uuid',
          artifact_path: 'somePath',
          utc_time_created: '2020-10-30',
          flavors: { keras: {}, python_function: {} },
        },
        {
          run_id: 'run-uuid',
          artifact_path: 'someOtherPath',
          utc_time_created: '2020-10-31',
          flavors: { sklearn: {}, python_function: {} },
        },
      ]),
    }),
  };
  const loggedModels = Utils.getLoggedModelsFromTags(tags);
  // Both registered - newer timestamp first
  let modelVersions = [
    {
      name: 'someModel',
      version: '3',
      source: 'nananaBatman/artifacts/somePath',
      run_id: 'run-uuid',
      creation_timestamp: 12345,
    },
    {
      name: 'someNewerModel',
      version: '4',
      source: 'nananaBatman/artifacts/someOtherPath',
      run_id: 'run-uuid',
      creation_timestamp: 67890,
    },
  ];
  let models = Utils.mergeLoggedAndRegisteredModels(loggedModels, modelVersions);
  expect(models.length).toEqual(2);
  expect(models[0]).toEqual({
    artifactPath: 'someOtherPath',
    flavors: ['sklearn'],
    utcTimeCreated: 1604102400,
    registeredModelName: 'someNewerModel',
    registeredModelVersion: '4',
    registeredModelCreationTimestamp: 67890,
  });
  expect(models[1]).toEqual({
    artifactPath: 'somePath',
    flavors: ['keras'],
    utcTimeCreated: 1604016000,
    registeredModelName: 'someModel',
    registeredModelVersion: '3',
    registeredModelCreationTimestamp: 12345,
  });
  // Change order
  modelVersions[1].name = 'someModel';
  modelVersions[1].version = '2';
  modelVersions[1].creation_timestamp = 1000;
  models = Utils.mergeLoggedAndRegisteredModels(loggedModels, modelVersions);
  expect(models[0].registeredModelVersion).toEqual('3');
  // Only one registered
  modelVersions = [modelVersions[0]];
  models = Utils.mergeLoggedAndRegisteredModels(loggedModels, modelVersions);
  expect(models[0]).toEqual({
    artifactPath: 'somePath',
    flavors: ['keras'],
    utcTimeCreated: 1604016000,
    registeredModelName: 'someModel',
    registeredModelVersion: '3',
    registeredModelCreationTimestamp: 12345,
  });
  expect(models[1]).toEqual({
    artifactPath: 'someOtherPath',
    flavors: ['sklearn'],
    utcTimeCreated: 1604102400,
  });
  // No registered; newest logged model first
  modelVersions = [];
  models = Utils.mergeLoggedAndRegisteredModels(loggedModels, modelVersions);
  expect(models[0]).toEqual({
    artifactPath: 'someOtherPath',
    flavors: ['sklearn'],
    utcTimeCreated: 1604102400,
  });
  expect(models[1]).toEqual({
    artifactPath: 'somePath',
    flavors: ['keras'],
    utcTimeCreated: 1604016000,
  });
});

test('concatAndGroupArraysById', () => {
  let arr: any;
  let concatArr: any;

  // sanity test
  arr = [];
  concatArr = [];
  expect(Utils.concatAndGroupArraysById(arr, concatArr, '')).toEqual([]);

  // basic functionality
  arr = [
    { name: 'harry', house: 'gryffindor', wand: 'holly' },
    { name: 'luna', house: 'ravenclaw', wand: 'unknown' },
    { name: 'draco', house: 'slytherin', wand: 'hawthorne' },
  ];
  concatArr = [
    { name: 'harry', enemy: 'voldemort' },
    { name: 'draco', enemy: 'harry' },
  ];
  expect(Utils.concatAndGroupArraysById(arr, concatArr, 'name')).toEqual([
    { name: 'harry', house: 'gryffindor', wand: 'holly', enemy: 'voldemort' },
    { name: 'luna', house: 'ravenclaw', wand: 'unknown' },
    { name: 'draco', house: 'slytherin', wand: 'hawthorne', enemy: 'harry' },
  ]);

  // no common ids - just concatenate
  arr = [
    { name: 'harry', house: 'gryffindor', wand: 'holly' },
    { name: 'luna', house: 'ravenclaw', wand: 'unknown' },
    { name: 'draco', house: 'slytherin', wand: 'hawthorne' },
  ];
  concatArr = [
    { name: 'ron', enemy: 'spiders' },
    { name: 'hermione', enemy: 'unknown' },
  ];
  expect(Utils.concatAndGroupArraysById(arr, concatArr, 'name')).toEqual([
    { name: 'harry', house: 'gryffindor', wand: 'holly' },
    { name: 'luna', house: 'ravenclaw', wand: 'unknown' },
    { name: 'draco', house: 'slytherin', wand: 'hawthorne' },
    { name: 'ron', enemy: 'spiders' },
    { name: 'hermione', enemy: 'unknown' },
  ]);

  // one common id, no additional fields
  arr = [{ name: 'harry', house: 'gryffindor', wand: 'holly' }];
  concatArr = [{ name: 'harry' }];
  expect(Utils.concatAndGroupArraysById(arr, concatArr, 'name')).toEqual([
    { name: 'harry', house: 'gryffindor', wand: 'holly' },
  ]);

  // different fields altogether
  arr = [{ name: 'harry', house: 'gryffindor', wand: 'holly' }];
  concatArr = [{ id: 123, year: 2020 }];
  expect(Utils.concatAndGroupArraysById(arr, concatArr, 'name')).toEqual([
    { name: 'harry', house: 'gryffindor', wand: 'holly' },
    { id: 123, year: 2020 },
  ]);
});

test('isValidHttpUrl', () => {
  expect(Utils.isValidHttpUrl('not_a_url')).toEqual(false);
  expect(Utils.isValidHttpUrl('https://some_url.com')).toEqual(true);
  expect(Utils.isValidHttpUrl('https://some_url.com/?with=query_param')).toEqual(true);
  expect(Utils.isValidHttpUrl('http://some_url.com/')).toEqual(true);
  expect(Utils.isValidHttpUrl('http:/localhost')).toEqual(true);
  expect(Utils.isValidHttpUrl('http:/a')).toEqual(true);
  expect(Utils.isValidHttpUrl('some other text https://some_url.com/')).toEqual(false);
  expect(Utils.isValidHttpUrl('fualksdhjakjsdh')).toEqual(false);
  expect(Utils.isValidHttpUrl('#')).toEqual(false);
  // disable no-script-url rule to test that javascript protocol is not seen as valid url
  /* eslint-disable no-script-url*/
  expect(Utils.isValidHttpUrl('javascript:void(0)')).toEqual(false);
});
