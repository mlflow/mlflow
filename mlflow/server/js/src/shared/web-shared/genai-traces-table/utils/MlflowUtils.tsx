/**
 * This file is a subset of functions from mlflow/web/js/src/common/Utils.tsx
 */
import type { TraceInfoV3 } from '../types';

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
class MlflowUtils {
  static runNameTag = 'mlflow.runName';
  static sourceNameTag = 'mlflow.source.name';
  static sourceTypeTag = 'mlflow.source.type';
  static entryPointTag = 'mlflow.project.entryPoint';

  static getEntryPointName(runTags: any) {
    const entryPointTag = runTags[MlflowUtils.entryPointTag];
    if (entryPointTag) {
      return entryPointTag.value;
    }
    return '';
  }

  static getSourceType(runTags: any) {
    const sourceTypeTag = runTags[MlflowUtils.sourceTypeTag];
    if (sourceTypeTag) {
      return sourceTypeTag.value;
    }
    return '';
  }

  static dropExtension(path: any) {
    return path.replace(/(.*[^/])\.[^/.]+$/, '$1');
  }

  static baseName(path: any) {
    const pieces = path.split('/');
    return pieces[pieces.length - 1];
  }

  /**
   * Renders the source name and entry point into a string. Used for sorting.
   * @param run MlflowMessages.RunInfo
   */
  static formatSource(tags: any) {
    const sourceName = MlflowUtils.getSourceName(tags);
    const sourceType = MlflowUtils.getSourceType(tags);
    const entryPointName = MlflowUtils.getEntryPointName(tags);
    if (sourceType === 'PROJECT') {
      let res = MlflowUtils.dropExtension(MlflowUtils.baseName(sourceName));
      if (entryPointName && entryPointName !== 'main') {
        res += ':' + entryPointName;
      }
      return res;
    } else if (sourceType === 'JOB') {
      const jobIdTag = 'mlflow.databricks.jobID';
      const jobRunIdTag = 'mlflow.databricks.jobRunID';
      const jobId = tags && tags[jobIdTag] && tags[jobIdTag].value;
      const jobRunId = tags && tags[jobRunIdTag] && tags[jobRunIdTag].value;
      if (jobId && jobRunId) {
        return MlflowUtils.getDefaultJobRunName(jobId, jobRunId);
      }
      return sourceName;
    } else {
      return MlflowUtils.baseName(sourceName);
    }
  }

  static getDefaultJobRunName(jobId: any, runId: any, workspaceId = null) {
    if (!jobId) {
      return '-';
    }
    let name = `job ${jobId}`;
    if (runId) {
      name = `run ${runId} of ` + name;
    }
    if (workspaceId) {
      name = `workspace ${workspaceId}: ` + name;
    }
    return name;
  }

  static getSourceName(runTags: any) {
    const sourceNameTag = runTags[MlflowUtils.sourceNameTag];
    if (sourceNameTag) {
      return sourceNameTag.value;
    }
    return '';
  }

  static getGitHubRegex() {
    return /[@/]github.com[:/]([^/.]+)\/([^/#]+)#?(.*)/;
  }

  static getGitLabRegex() {
    return /[@/]gitlab.com[:/]([^/.]+)\/([^/#]+)#?(.*)/;
  }

  static getBitbucketRegex() {
    return /[@/]bitbucket.org[:/]([^/.]+)\/([^/#]+)#?(.*)/;
  }

  static getRunPageRoute(experimentId: string, runUuid: string) {
    return `/experiments/${experimentId}/runs/${runUuid}`;
  }

  static getLoggedModelPageRoute(experimentId: string, loggedModelId: string) {
    return `/experiments/${experimentId}/models/${loggedModelId}`;
  }

  /**
   * Regular expression for URLs containing the string 'git'.
   * It can be a custom git domain (e.g. https://git.custom.in/repo/dir#file/dir).
   * Excluding the first overall match, there are three groups:
   *    git url, repo directory, and file directory.
   * (e.g. group1: https://custom.git.domain, group2: repo/directory, group3: project/directory)
   */
  static getGitRegex() {
    return /(.*?[@/][^?]*git.*?)[:/]([^#]+)(?:#(.*))?/;
  }

  static getGitRepoUrl(sourceName: any, branchName = 'master') {
    const gitHubMatch = sourceName.match(MlflowUtils.getGitHubRegex());
    const gitLabMatch = sourceName.match(MlflowUtils.getGitLabRegex());
    const bitbucketMatch = sourceName.match(MlflowUtils.getBitbucketRegex());
    const gitMatch = sourceName.match(MlflowUtils.getGitRegex());
    let url = null;
    if (gitHubMatch) {
      url = `https://github.com/${gitHubMatch[1]}/${gitHubMatch[2].replace(/.git/, '')}`;
      if (gitHubMatch[3]) {
        url += `/tree/${branchName}/${gitHubMatch[3]}`;
      }
    } else if (gitLabMatch) {
      url = `https://gitlab.com/${gitLabMatch[1]}/${gitLabMatch[2].replace(/.git/, '')}`;
      if (gitLabMatch[3]) {
        url += `/-/tree/${branchName}/${gitLabMatch[3]}`;
      }
    } else if (bitbucketMatch) {
      url = `https://bitbucket.org/${bitbucketMatch[1]}/${bitbucketMatch[2].replace(/.git/, '')}`;
      if (bitbucketMatch[3]) {
        url += `/src/${branchName}/${bitbucketMatch[3]}`;
      }
    } else if (gitMatch) {
      const [, baseUrl, repoDir, fileDir] = gitMatch;
      url = baseUrl.replace(/git@/, 'https://') + '/' + repoDir.replace(/.git/, '');
      if (fileDir) {
        url += `/tree/${branchName}/${fileDir}`;
      }
    }
    return url;
  }

  static getNotebookRevisionId(tags: any) {
    const revisionIdTag = 'mlflow.databricks.notebookRevisionID';
    return tags && tags[revisionIdTag] && tags[revisionIdTag].value;
  }

  static getNotebookId(tags: any) {
    const notebookIdTag = 'mlflow.databricks.notebookID';
    return tags && tags[notebookIdTag] && tags[notebookIdTag].value;
  }

  /**
   * Check if the given workspaceId matches the current workspaceId.
   * @param workspaceId
   * @returns {boolean}
   */
  static isCurrentWorkspace(workspaceId: any) {
    return true;
  }

  static getDefaultNotebookRevisionName(notebookId: any, revisionId: any, workspaceId = null) {
    if (!notebookId) {
      return '-';
    }
    let name = `notebook ${notebookId}`;
    if (revisionId) {
      name = `revision ${revisionId} of ` + name;
    }
    if (workspaceId) {
      name = `workspace ${workspaceId}: ` + name;
    }
    return name;
  }

  /**
   * Makes sure that the URL begins with correct scheme according
   * to RFC3986 [https://datatracker.ietf.org/doc/html/rfc3986#section-3.1]
   * It does not support slash-less schemes (e.g. news:abc, urn:anc).
   * @param url URL string like "my-mlflow-server.com/#/experiments/9" or
   *        "https://my-mlflow-server.com/#/experiments/9"
   * @param defaultScheme scheme to add if missing in the provided URL, defaults to "https"
   * @returns {string} the URL string with ensured default scheme
   */
  static ensureUrlScheme(url: any, defaultScheme = 'https') {
    // Falsy values should yield itself
    if (!url) return url;

    // Scheme-less URL with colon and dashes
    if (url.match(/^:\/\//i)) {
      return `${defaultScheme}${url}`;
    }

    // URL without scheme, colon nor dashes
    if (!url.match(/^[a-z1-9+-.]+:\/\//i)) {
      return `${defaultScheme}://${url}`;
    }

    // Pass-through for "correct" entries
    return url;
  }

  /**
   * Returns a copy of the provided URL with its query parameters set to `queryParams`.
   * @param url URL string like "http://my-mlflow-server.com/#/experiments/9.
   * @param queryParams Optional query parameter string like "?param=12345". Query params provided
   *        via this string will override existing query param values in `url`
   */
  static setQueryParams(url: any, queryParams: any) {
    // Using new URL() is the preferred way of constructing the URL object,
    // however according to [https://url.spec.whatwg.org/#constructors] it requires
    // providing the protocol. We're gracefully ensuring that the scheme exists here.
    const urlObj = new URL(MlflowUtils.ensureUrlScheme(url));
    urlObj.search = queryParams || '';
    return urlObj.toString();
  }

  /**
   * Returns the URL for the notebook source.
   */
  static getNotebookSourceUrl(queryParams: any, notebookId: any, revisionId: any, runUuid: any, workspaceUrl = null) {
    let url = MlflowUtils.setQueryParams(workspaceUrl || window.location.origin, queryParams);
    url += `#notebook/${notebookId}`;
    if (revisionId) {
      url += `/revision/${revisionId}`;
      if (runUuid) {
        url += `/mlflow/run/${runUuid}`;
      }
    }
    return url;
  }

  /**
   * Renders the notebook source name and entry point into an HTML element. Used for display.
   */
  static renderNotebookSource(
    queryParams: any,
    notebookId: any,
    revisionId: any,
    runUuid: any,
    sourceName: any,
    workspaceUrl = null,
    nameOverride: string | null = null,
  ) {
    // sourceName may not be present when rendering feature table notebook consumers from remote
    // workspaces or when notebook fetcher failed to fetch the sourceName. Always provide a default
    // notebook name in such case.
    const baseName = sourceName
      ? MlflowUtils.baseName(sourceName)
      : MlflowUtils.getDefaultNotebookRevisionName(notebookId, revisionId);
    const name = nameOverride || baseName;

    if (notebookId) {
      const url = MlflowUtils.getNotebookSourceUrl(queryParams, notebookId, revisionId, runUuid, workspaceUrl);
      return (
        <a
          title={sourceName || MlflowUtils.getDefaultNotebookRevisionName(notebookId, revisionId)}
          href={url}
          target="_top"
        >
          {name}
        </a>
      );
    } else {
      return name;
    }
  }

  /**
   * Set query params and returns the updated query params.
   * @returns {string} updated query params
   */
  static addQueryParams(currentQueryParams: any, newQueryParams: any) {
    if (!newQueryParams || Object.keys(newQueryParams).length === 0) {
      return currentQueryParams;
    }
    const urlSearchParams = new URLSearchParams(currentQueryParams);
    Object.entries(newQueryParams).forEach(
      // @ts-expect-error TS(2345): Argument of type 'unknown' is not assignable to pa... Remove this comment to see the full error message
      ([key, value]) => Boolean(key) && Boolean(value) && urlSearchParams.set(key, value),
    );
    const queryParams = urlSearchParams.toString();
    if (queryParams !== '' && !queryParams.includes('?')) {
      return `?${queryParams}`;
    }
    return queryParams;
  }

  /**
   * Returns the URL for the job source.
   */
  static getJobSourceUrl(queryParams: any, jobId: any, jobRunId: any, workspaceUrl = null) {
    let url = MlflowUtils.setQueryParams(workspaceUrl || window.location.origin, queryParams);
    url += `#job/${jobId}`;
    if (jobRunId) {
      url += `/run/${jobRunId}`;
    }
    return url;
  }

  /**
   * Renders the job source name and entry point into an HTML element. Used for display.
   */
  static renderJobSource(
    queryParams: any,
    jobId: any,
    jobRunId: any,
    jobName: any,
    workspaceUrl = null,
    nameOverride: string | null = null,
  ) {
    // jobName may not be present when rendering feature table job consumers from remote
    // workspaces or when getJob API failed to fetch the jobName. Always provide a default
    // job name in such case.
    const reformatJobName = jobName || MlflowUtils.getDefaultJobRunName(jobId, jobRunId);
    const name = nameOverride || reformatJobName;

    if (jobId) {
      const url = MlflowUtils.getJobSourceUrl(queryParams, jobId, jobRunId, workspaceUrl);
      return (
        <a title={reformatJobName} href={url} target="_top">
          {name}
        </a>
      );
    } else {
      return name;
    }
  }

  /**
   * Renders the source name and entry point into an HTML element. Used for display.
   * @param tags Object containing tag key value pairs.
   * @param queryParams Query params to add to certain source type links.
   * @param runUuid ID of the MLflow run to add to certain source (revision) links.
   */
  static renderSource(tags: any, queryParams: any, runUuid: any, branchName = 'master') {
    const sourceName = MlflowUtils.getSourceName(tags);
    let res = MlflowUtils.formatSource(tags);
    const gitRepoUrlOrNull = MlflowUtils.getGitRepoUrl(sourceName, branchName);
    if (gitRepoUrlOrNull) {
      res = (
        <a target="_top" href={gitRepoUrlOrNull}>
          {res}
        </a>
      );
    }
    const sourceType = MlflowUtils.getSourceType(tags);
    if (sourceType === 'NOTEBOOK') {
      const revisionId = MlflowUtils.getNotebookRevisionId(tags);
      const notebookId = MlflowUtils.getNotebookId(tags);
      const workspaceIdTag = 'mlflow.databricks.workspaceID';
      const workspaceId = tags && tags[workspaceIdTag] && tags[workspaceIdTag].value;
      if (MlflowUtils.isCurrentWorkspace(workspaceId)) {
        return MlflowUtils.renderNotebookSource(queryParams, notebookId, revisionId, runUuid, sourceName, null);
      } else {
        const workspaceUrlTag = 'mlflow.databricks.workspaceURL';
        const workspaceUrl = tags && tags[workspaceUrlTag] && tags[workspaceUrlTag].value;
        const notebookQueryParams = MlflowUtils.addQueryParams(queryParams, { o: workspaceId });
        return MlflowUtils.renderNotebookSource(
          notebookQueryParams,
          notebookId,
          revisionId,
          runUuid,
          sourceName,
          workspaceUrl,
        );
      }
    }
    if (sourceType === 'JOB') {
      const jobIdTag = 'mlflow.databricks.jobID';
      const jobRunIdTag = 'mlflow.databricks.jobRunID';
      const jobId = tags && tags[jobIdTag] && tags[jobIdTag].value;
      const jobRunId = tags && tags[jobRunIdTag] && tags[jobRunIdTag].value;
      const workspaceIdTag = 'mlflow.databricks.workspaceID';
      const workspaceId = tags && tags[workspaceIdTag] && tags[workspaceIdTag].value;
      if (MlflowUtils.isCurrentWorkspace(workspaceId)) {
        return MlflowUtils.renderJobSource(queryParams, jobId, jobRunId, res, null);
      } else {
        const workspaceUrlTag = 'mlflow.databricks.workspaceURL';
        const workspaceUrl = tags && tags[workspaceUrlTag] && tags[workspaceUrlTag].value;
        const jobQueryParams = MlflowUtils.addQueryParams(queryParams, { o: workspaceId });
        return MlflowUtils.renderJobSource(jobQueryParams, jobId, jobRunId, res, workspaceUrl);
      }
    }
    return res;
  }

  static renderSourceFromMetadata(traceInfoV3: TraceInfoV3) {
    const sourceName = traceInfoV3.trace_metadata?.[MlflowUtils.sourceNameTag];
    const sourceType = traceInfoV3.trace_metadata?.[MlflowUtils.sourceTypeTag];
    let res = sourceName ? MlflowUtils.baseName(sourceName) : '';

    // Handle git repository links using explicit git metadata
    const gitRepoUrl = traceInfoV3.trace_metadata?.['mlflow.source.git.repoURL'];
    const gitBranch = traceInfoV3.trace_metadata?.['mlflow.source.git.branch'];
    const gitCommit = traceInfoV3.trace_metadata?.['mlflow.source.git.commit'];

    if (gitRepoUrl) {
      // Convert SSH URL to HTTPS if needed
      const httpsUrl = gitRepoUrl
        .replace('git@github.com:', 'https://github.com/')
        .replace('git@gitlab.com:', 'https://gitlab.com/')
        .replace('git@bitbucket.org:', 'https://bitbucket.org/')
        .replace('.git', '');

      // Use commit hash if available, otherwise use branch
      const ref = gitCommit || gitBranch || 'master';
      const filePath = sourceName ? `/${sourceName}` : '';

      // Construct URL based on the git host
      let url = httpsUrl;
      if (httpsUrl.includes('github.com')) {
        url = `${httpsUrl}/tree/${ref}${filePath}`;
      } else if (httpsUrl.includes('gitlab.com')) {
        url = `${httpsUrl}/-/tree/${ref}${filePath}`;
      } else if (httpsUrl.includes('bitbucket.org')) {
        url = `${httpsUrl}/src/${ref}${filePath}`;
      } else {
        // For other git hosts, just append the ref and file path
        url = `${httpsUrl}/tree/${ref}${filePath}`;
      }

      res = (
        <a target="_blank" rel="noopener noreferrer" href={url}>
          {res}
        </a>
      );
    }

    if (sourceType === 'NOTEBOOK') {
      const revisionId = traceInfoV3.trace_metadata?.['mlflow.databricks.notebookRevisionID'];
      const notebookId = traceInfoV3.trace_metadata?.['mlflow.databricks.notebookID'];
      const workspaceId = traceInfoV3.trace_metadata?.['mlflow.databricks.workspaceID'];

      if (MlflowUtils.isCurrentWorkspace(workspaceId)) {
        return MlflowUtils.renderNotebookSource(null, notebookId, revisionId, null, sourceName, null);
      } else {
        const workspaceUrlTag = 'mlflow.databricks.workspaceURL';
        const workspaceUrl: any = traceInfoV3.trace_metadata?.[workspaceUrlTag] || undefined;
        const notebookQueryParams = MlflowUtils.addQueryParams(null, { o: workspaceId });
        return MlflowUtils.renderNotebookSource(
          notebookQueryParams,
          notebookId,
          revisionId,
          null,
          sourceName,
          workspaceUrl,
        );
      }
    }

    if (sourceType === 'JOB') {
      const jobId = traceInfoV3.trace_metadata?.['mlflow.databricks.jobID'];
      const jobRunId = traceInfoV3.trace_metadata?.['mlflow.databricks.jobRunID'];
      const workspaceId = traceInfoV3.trace_metadata?.['mlflow.databricks.workspaceID'];

      if (MlflowUtils.isCurrentWorkspace(workspaceId)) {
        return MlflowUtils.renderJobSource(null, jobId, jobRunId, res, null);
      } else {
        const workspaceUrlTag = 'mlflow.databricks.workspaceURL';
        const workspaceUrl: any = traceInfoV3.trace_metadata?.[workspaceUrlTag] || undefined;
        const jobQueryParams = MlflowUtils.addQueryParams(null, { o: workspaceId });
        return MlflowUtils.renderJobSource(jobQueryParams, jobId, jobRunId, res, workspaceUrl);
      }
    }

    return res;
  }
}

export default MlflowUtils;
