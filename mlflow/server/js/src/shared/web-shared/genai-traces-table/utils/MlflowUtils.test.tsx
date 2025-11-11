/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import React from 'react';

import MlflowUtils from './MlflowUtils';

test('baseName', () => {
  expect(MlflowUtils.baseName('foo')).toEqual('foo');
  expect(MlflowUtils.baseName('foo/bar/baz')).toEqual('baz');
  expect(MlflowUtils.baseName('/foo/bar/baz')).toEqual('baz');
  expect(MlflowUtils.baseName('file:///foo/bar/baz')).toEqual('baz');
});

test('renderNotebookSource', () => {
  const notebookId = '12345678';
  const revisionId = '987654';
  const runUuid = '1133557799';
  const sourceName = '/Users/test/iris_feature';
  const nameOverride = 'some feature';
  const queryParams = '?o=123456789';

  expect(MlflowUtils.renderNotebookSource(null, null, null, null, sourceName, null)).toEqual('iris_feature');
  expect(MlflowUtils.renderNotebookSource(null, notebookId, null, null, sourceName, null)).toEqual(
    <a title={sourceName} href={`http://localhost/#notebook/${notebookId}`} target="_top">
      iris_feature
    </a>,
  );
  expect(MlflowUtils.renderNotebookSource(null, notebookId, revisionId, null, sourceName, null)).toEqual(
    <a title={sourceName} href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}`} target="_top">
      iris_feature
    </a>,
  );
  expect(MlflowUtils.renderNotebookSource(null, notebookId, revisionId, runUuid, sourceName, null)).toEqual(
    <a
      title={sourceName}
      href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      iris_feature
    </a>,
  );
  expect(MlflowUtils.renderNotebookSource(null, notebookId, revisionId, runUuid, null, null)).toEqual(
    <a
      title={MlflowUtils.getDefaultNotebookRevisionName(notebookId, revisionId)}
      href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      {MlflowUtils.getDefaultNotebookRevisionName(notebookId, revisionId)}
    </a>,
  );
  expect(
    MlflowUtils.renderNotebookSource(null, notebookId, revisionId, runUuid, sourceName, null, nameOverride),
  ).toEqual(
    <a
      title={sourceName}
      href={`http://localhost/#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      {nameOverride}
    </a>,
  );
  expect(MlflowUtils.renderNotebookSource(queryParams, notebookId, revisionId, runUuid, sourceName, null)).toEqual(
    <a
      title={sourceName}
      href={`http://localhost/${queryParams}#notebook/${notebookId}/revision/${revisionId}/mlflow/run/${runUuid}`}
      target="_top"
    >
      iris_feature
    </a>,
  );
  expect(
    MlflowUtils.renderNotebookSource(
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

  expect(MlflowUtils.renderJobSource(null, null, null, jobName, null)).toEqual(jobName);
  expect(MlflowUtils.renderJobSource(null, jobId, null, jobName, null)).toEqual(
    <a title={jobName} href={`http://localhost/#job/${jobId}`} target="_top">
      {jobName}
    </a>,
  );
  expect(MlflowUtils.renderJobSource(null, jobId, null, null, null)).toEqual(
    <a title={`job ${jobId}`} href={`http://localhost/#job/${jobId}`} target="_top">
      {`job ${jobId}`}
    </a>,
  );
  expect(MlflowUtils.renderJobSource(null, jobId, jobRunId, jobName, null)).toEqual(
    <a title={jobName} href={`http://localhost/#job/${jobId}/run/${jobRunId}`} target="_top">
      {jobName}
    </a>,
  );
  expect(MlflowUtils.renderJobSource(null, jobId, jobRunId, null, null)).toEqual(
    <a
      title={MlflowUtils.getDefaultJobRunName(jobId, jobRunId)}
      href={`http://localhost/#job/${jobId}/run/${jobRunId}`}
      target="_top"
    >
      {MlflowUtils.getDefaultJobRunName(jobId, jobRunId)}
    </a>,
  );
  expect(MlflowUtils.renderJobSource(null, jobId, jobRunId, jobName, null, nameOverride)).toEqual(
    <a title={jobName} href={`http://localhost/#job/${jobId}/run/${jobRunId}`} target="_top">
      {nameOverride}
    </a>,
  );
  expect(MlflowUtils.renderJobSource(queryParams, jobId, jobRunId, jobName, null)).toEqual(
    <a title={jobName} href={`http://localhost/${queryParams}#job/${jobId}/run/${jobRunId}`} target="_top">
      {jobName}
    </a>,
  );
  expect(
    // @ts-expect-error TS(2345): Argument of type '"https://databricks"' is not ass... Remove this comment to see the full error message
    MlflowUtils.renderJobSource(queryParams, jobId, jobRunId, jobName, 'https://databricks', null),
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
  expect(MlflowUtils.formatSource(source_with_name)).toEqual('source:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(source_with_name)).toEqual('source:entry');

  const source_with_main = {
    'mlflow.source.name': { value: 'source1' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'main' },
  };
  expect(MlflowUtils.formatSource(source_with_main)).toEqual('source1');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(source_with_main)).toEqual('source1');

  const source_no_name = {
    'mlflow.source.name': { value: 'source2' },
    'mlflow.source.type': { value: 'PROJECT' },
  };
  expect(MlflowUtils.formatSource(source_no_name)).toEqual('source2');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(source_no_name)).toEqual('source2');

  const non_project_source = {
    'mlflow.source.name': { value: 'source3' },
    'mlflow.source.type': { value: 'NOTEBOOK' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(MlflowUtils.formatSource(non_project_source)).toEqual('source3');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(non_project_source)).toEqual('source3');

  // formatSource should return a string, renderSource should return an HTML element.
  const github_url = {
    'mlflow.source.name': { value: 'git@github.com:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(MlflowUtils.formatSource(github_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(github_url)).toEqual(
    <a href="https://github.com/mlflow/mlflow-apps" target="_top">
      mlflow-apps:entry
    </a>,
  );

  const gitlab_url = {
    'mlflow.source.name': { value: 'git@gitlab.com:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(MlflowUtils.formatSource(gitlab_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(gitlab_url)).toEqual(
    <a href="https://gitlab.com/mlflow/mlflow-apps" target="_top">
      mlflow-apps:entry
    </a>,
  );

  const gitlab_long_url = {
    'mlflow.source.name': { value: 'git@gitlab.com:mlflow/mlflow-apps.git#tmp' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(MlflowUtils.formatSource(gitlab_long_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(gitlab_long_url)).toEqual(
    <a href="https://gitlab.com/mlflow/mlflow-apps/-/tree/master/tmp" target="_top">
      mlflow-apps:entry
    </a>,
  );

  const bitbucket_url = {
    'mlflow.source.name': { value: 'git@bitbucket.org:mlflow/mlflow-apps.git' },
    'mlflow.source.type': { value: 'PROJECT' },
    'mlflow.project.entryPoint': { value: 'entry' },
  };
  expect(MlflowUtils.formatSource(bitbucket_url)).toEqual('mlflow-apps:entry');
  // @ts-expect-error TS(2554): Expected 3 arguments, but got 1.
  expect(MlflowUtils.renderSource(bitbucket_url)).toEqual(
    <a href="https://bitbucket.org/mlflow/mlflow-apps" target="_top">
      mlflow-apps:entry
    </a>,
  );
});

test('setQueryParams', () => {
  expect(MlflowUtils.setQueryParams('http://localhost/foo', '?o=123')).toEqual('http://localhost/foo?o=123');
  expect(MlflowUtils.setQueryParams('http://localhost/foo?param=val', '?o=123')).toEqual('http://localhost/foo?o=123');
  expect(MlflowUtils.setQueryParams('http://localhost/foo?param=val', '?param=newval')).toEqual(
    'http://localhost/foo?param=newval',
  );
  expect(MlflowUtils.setQueryParams('https://localhost/foo?param=val', '?param=newval')).toEqual(
    'https://localhost/foo?param=newval',
  );
  expect(MlflowUtils.setQueryParams('localhost/foo?param=val', '?param=newval')).toEqual(
    'https://localhost/foo?param=newval',
  );
});

test('ensureUrlScheme', () => {
  expect(MlflowUtils.ensureUrlScheme('http://localhost/xyz/abc?o=123')).toEqual('http://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('https://localhost/xyz/abc?o=123')).toEqual('https://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('HTTPS://localhost/xyz/abc?o=123')).toEqual('HTTPS://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('localhost/xyz/abc?o=123')).toEqual('https://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('localhost/xyz/abc?o=123', 'http')).toEqual('http://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('user:pass@localhost/xyz/abc?o=123')).toEqual(
    'https://user:pass@localhost/xyz/abc?o=123',
  );
  expect(MlflowUtils.ensureUrlScheme('https://user:pass@localhost/xyz/abc?o=123')).toEqual(
    'https://user:pass@localhost/xyz/abc?o=123',
  );
  expect(MlflowUtils.ensureUrlScheme('https://localhost/xyz/abc?o=123', 'http')).toEqual(
    'https://localhost/xyz/abc?o=123',
  );
  expect(MlflowUtils.ensureUrlScheme('://localhost/xyz/abc?o=123', 'https')).toEqual('https://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('://localhost/xyz/abc?o=123', 'ws')).toEqual('ws://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('wss://localhost/xyz/abc?o=123')).toEqual('wss://localhost/xyz/abc?o=123');
  expect(MlflowUtils.ensureUrlScheme('scheme-with+symbols.123x://localhost/xyz/abc?o=123')).toEqual(
    'scheme-with+symbols.123x://localhost/xyz/abc?o=123',
  );
  expect(MlflowUtils.ensureUrlScheme('legal-schema://abc')).toEqual('legal-schema://abc');
  expect(MlflowUtils.ensureUrlScheme('illegal_schema://abc')).toEqual('https://illegal_schema://abc');
  expect(MlflowUtils.ensureUrlScheme(undefined)).toEqual(undefined);
});

test('addQueryParams', () => {
  expect(MlflowUtils.addQueryParams('', { o: null })).toEqual('');
  expect(MlflowUtils.addQueryParams('?param=val', { o: null })).toEqual('?param=val');
  expect(MlflowUtils.addQueryParams('', { o: 123 })).toEqual('?o=123');
  expect(MlflowUtils.addQueryParams('', { o: 123, param: 'val' })).toEqual('?o=123&param=val');
  expect(MlflowUtils.addQueryParams('?param=val', { o: 123 })).toEqual('?param=val&o=123');
  expect(MlflowUtils.addQueryParams('?o=456', { o: 123 })).toEqual('?o=123');
});

test('getDefaultJobRunName', () => {
  expect(MlflowUtils.getDefaultJobRunName(null, null)).toEqual('-');
  expect(MlflowUtils.getDefaultJobRunName(123, null)).toEqual('job 123');
  expect(MlflowUtils.getDefaultJobRunName(123, 456)).toEqual('run 456 of job 123');
  // @ts-expect-error TS(2345): Argument of type '7890' is not assignable to param... Remove this comment to see the full error message
  expect(MlflowUtils.getDefaultJobRunName(123, 456, 7890)).toEqual('workspace 7890: run 456 of job 123');
});

test('getDefaultNotebookRevisionName', () => {
  expect(MlflowUtils.getDefaultNotebookRevisionName(null, null)).toEqual('-');
  expect(MlflowUtils.getDefaultNotebookRevisionName(123, null)).toEqual('notebook 123');
  expect(MlflowUtils.getDefaultNotebookRevisionName(123, 456)).toEqual('revision 456 of notebook 123');
  // @ts-expect-error TS(2345): Argument of type '7890' is not assignable to param... Remove this comment to see the full error message
  expect(MlflowUtils.getDefaultNotebookRevisionName(123, 456, 7890)).toEqual(
    'workspace 7890: revision 456 of notebook 123',
  );
});

test('dropExtension', () => {
  expect(MlflowUtils.dropExtension('foo')).toEqual('foo');
  expect(MlflowUtils.dropExtension('foo.xyz')).toEqual('foo');
  expect(MlflowUtils.dropExtension('foo.xyz.zyx')).toEqual('foo.xyz');
  expect(MlflowUtils.dropExtension('foo/bar/baz.xyz')).toEqual('foo/bar/baz');
  expect(MlflowUtils.dropExtension('.foo/.bar/baz.xyz')).toEqual('.foo/.bar/baz');
  expect(MlflowUtils.dropExtension('.foo')).toEqual('.foo');
  expect(MlflowUtils.dropExtension('.foo.bar')).toEqual('.foo');
  expect(MlflowUtils.dropExtension('/.foo')).toEqual('/.foo');
  expect(MlflowUtils.dropExtension('.foo/.bar/.xyz')).toEqual('.foo/.bar/.xyz');
});

test('getGitHubRegex', () => {
  const gitHubRegex = MlflowUtils.getGitHubRegex();
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
  const gitLabRegex = MlflowUtils.getGitLabRegex();
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
  const gitRegex = MlflowUtils.getGitRegex();
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

test('renderSourceFromMetadata', () => {
  // Test git repository with commit
  const gitSourceWithCommit: any = {
    trace_metadata: {
      'mlflow.source.name': 'runner.py',
      'mlflow.source.type': 'PROJECT',
      'mlflow.source.git.repoURL': 'git@github.com:username/repo.git',
      'mlflow.source.git.commit': 'b2fda8216f2568b62213ee99314f5bf4ce89be96',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(gitSourceWithCommit)).toEqual(
    <a
      target="_blank"
      rel="noopener noreferrer"
      href="https://github.com/username/repo/tree/b2fda8216f2568b62213ee99314f5bf4ce89be96/runner.py"
    >
      runner.py
    </a>,
  );

  // Test git repository with branch
  const gitSourceWithBranch: any = {
    trace_metadata: {
      'mlflow.source.name': 'runner.py',
      'mlflow.source.type': 'PROJECT',
      'mlflow.source.git.repoURL': 'git@github.com:username/repo.git',
      'mlflow.source.git.branch': 'main',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(gitSourceWithBranch)).toEqual(
    <a target="_blank" rel="noopener noreferrer" href="https://github.com/username/repo/tree/main/runner.py">
      runner.py
    </a>,
  );

  // Test git repository with GitLab
  const gitSourceGitLab: any = {
    trace_metadata: {
      'mlflow.source.name': 'runner.py',
      'mlflow.source.type': 'PROJECT',
      'mlflow.source.git.repoURL': 'git@gitlab.com:username/repo.git',
      'mlflow.source.git.commit': 'b2fda8216f2568b62213ee99314f5bf4ce89be96',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(gitSourceGitLab)).toEqual(
    <a
      target="_blank"
      rel="noopener noreferrer"
      href="https://gitlab.com/username/repo/-/tree/b2fda8216f2568b62213ee99314f5bf4ce89be96/runner.py"
    >
      runner.py
    </a>,
  );

  // Test notebook source
  const notebookSource: any = {
    trace_metadata: {
      'mlflow.source.name': 'notebook.py',
      'mlflow.source.type': 'NOTEBOOK',
      'mlflow.databricks.notebookID': '123456',
      'mlflow.databricks.notebookRevisionID': '987654',
      'mlflow.databricks.workspaceID': '0',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(notebookSource)).toEqual(
    <a title="notebook.py" href="http://localhost/#notebook/123456/revision/987654" target="_top">
      notebook.py
    </a>,
  );

  // Test job source
  const jobSource: any = {
    trace_metadata: {
      'mlflow.source.name': 'job.py',
      'mlflow.source.type': 'JOB',
      'mlflow.databricks.jobID': '123456',
      'mlflow.databricks.jobRunID': '987654',
      'mlflow.databricks.workspaceID': '0',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(jobSource)).toEqual(
    <a title="job.py" href="http://localhost/#job/123456/run/987654" target="_top">
      job.py
    </a>,
  );

  // Test source with no type
  const noTypeSource: any = {
    trace_metadata: {
      'mlflow.source.name': 'file.py',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(noTypeSource)).toEqual('file.py');

  // Test source with no name
  const noNameSource: any = {
    trace_metadata: {
      'mlflow.source.type': 'PROJECT',
    },
  };
  expect(MlflowUtils.renderSourceFromMetadata(noNameSource)).toEqual('');
});
