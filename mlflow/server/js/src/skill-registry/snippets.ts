// Copyable CLI and Python examples. Every user-supplied value goes through a quoting helper.
import { SkillStatus } from './types';
import type { SkillRegistrationSourceType } from './sourceLocation';
import type { TrackingSnippetConnection } from '../common/utils/trackingSnippetConnection';

const quoteShellArg = (value: string) => `'${value.replace(/'/g, "'\\''")}'`;

// JSON string escapes (\", \\, \n, \uXXXX) are all valid Python string-literal escapes.
const quotePythonString = (value: string) => JSON.stringify(value);

// The MLflow client reads the workspace from MLFLOW_WORKSPACE, so it scopes any `mlflow` command.
const cliCommand = (lines: string[], connection: TrackingSnippetConnection = {}) => {
  const env = [
    connection.trackingUri && `MLFLOW_TRACKING_URI=${quoteShellArg(connection.trackingUri)}`,
    connection.trackingAuth && `MLFLOW_TRACKING_AUTH=${quoteShellArg(connection.trackingAuth)}`,
    connection.workspace && `MLFLOW_WORKSPACE=${quoteShellArg(connection.workspace)}`,
  ].filter(Boolean);
  return [...env, ...lines.map((line, index) => (index > 0 ? `    ${line}` : line))].join(' \\\n');
};

const pythonCall = (call: string, args: string[]) => `${call}(\n${args.map((arg) => `    ${arg},`).join('\n')}\n)`;

const pythonSnippet = (imports: string[], call: string, connection: TrackingSnippetConnection = {}) => {
  const authEnv =
    connection.trackingAuth && `os.environ["MLFLOW_TRACKING_AUTH"] = ${quotePythonString(connection.trackingAuth)}`;
  return [
    ...(authEnv ? ['import os', '', authEnv, ''] : []),
    ...imports,
    '',
    ...(connection.trackingUri ? [`mlflow.set_tracking_uri(${quotePythonString(connection.trackingUri)})`] : []),
    ...(connection.workspace ? [`mlflow.set_workspace(${quotePythonString(connection.workspace)})`] : []),
    call,
  ].join('\n');
};

const PYTHON_SOURCE_CLASS: Record<SkillRegistrationSourceType, string> = {
  git: 'GitSource',
  oci: 'OCISource',
  zip: 'ZipSource',
};

export interface SkillRegisterSnippetOptions extends TrackingSnippetConnection {
  sourceType?: SkillRegistrationSourceType;
  location: string;
  local: boolean;
  name?: string;
  organization?: string;
  ref?: string;
  subpath?: string;
  status?: SkillStatus;
}

export const formatSkillRegisterCli = ({
  sourceType = 'git',
  location,
  local,
  name,
  organization,
  ref,
  subpath,
  status,
  workspace,
  trackingUri,
  trackingAuth,
}: SkillRegisterSnippetOptions) => {
  const lines = [
    local ? `mlflow skills register --source ${quoteShellArg('<directory>')}` : `mlflow skills register ${sourceType}`,
  ];
  if (name) lines.push(`--name ${quoteShellArg(name)}`);
  if (organization) lines.push(`--organization ${quoteShellArg(organization)}`);
  if (!local) {
    lines.push(`${sourceType === 'oci' ? '--image' : '--url'} ${quoteShellArg(location.trim() || '<location>')}`);
    if (ref && sourceType === 'git') lines.push(`--ref ${quoteShellArg(ref)}`);
    if (subpath) lines.push(`--subpath ${quoteShellArg(subpath)}`);
  }
  if (status && status !== SkillStatus.ACTIVE) lines.push(`--status ${status}`);
  return cliCommand(lines, { workspace, trackingUri, trackingAuth });
};

export const formatSkillRegisterPython = ({
  sourceType = 'git',
  location,
  local,
  name,
  organization,
  ref,
  subpath,
  status,
  workspace,
  trackingUri,
  trackingAuth,
}: SkillRegisterSnippetOptions) => {
  const args: string[] = [];
  if (name) args.push(`name=${quotePythonString(name)}`);
  if (organization) args.push(`organization=${quotePythonString(organization)}`);
  if (local) {
    args.push('source="<directory>"');
  } else {
    const sourceArgs = [
      `${sourceType === 'oci' ? 'image' : 'url'}=${quotePythonString(location.trim() || '<location>')}`,
    ];
    if (ref && sourceType === 'git') sourceArgs.push(`ref=${quotePythonString(ref)}`);
    if (subpath) sourceArgs.push(`subpath=${quotePythonString(subpath)}`);
    args.push(`source=${PYTHON_SOURCE_CLASS[sourceType]}(${sourceArgs.join(', ')})`);
  }
  if (status && status !== SkillStatus.ACTIVE) args.push(`status=${quotePythonString(status)}`);
  const imports = local
    ? ['import mlflow']
    : ['import mlflow', `from mlflow.genai import ${PYTHON_SOURCE_CLASS[sourceType]}`];
  return pythonSnippet(imports, pythonCall('mlflow.genai.register_skill', args), {
    workspace,
    trackingUri,
    trackingAuth,
  });
};

export interface SkillImportSnippetOptions extends TrackingSnippetConnection {
  source: string;
  ref?: string;
  /** Discovery root; the import registers every SKILL.md found beneath it. */
  subpath?: string;
  organization?: string;
  status?: SkillStatus;
}

/**
 * The examples behind the registration dialog. A new skill registered from Git also gets the repository import,
 * since the UI cannot see whether a Git location is one skill or a folder of skills, and the import registers
 * every SKILL.md beneath the path.
 */
export const buildRegistrationSnippets = ({
  register,
  repositoryUrl,
}: {
  register: SkillRegisterSnippetOptions;
  /** Set only for a new skill: adding a version to an existing skill has no repository import. */
  repositoryUrl?: string | null;
}): { register: SkillRegisterSnippetOptions; repositoryImport?: SkillImportSnippetOptions } => ({
  register,
  repositoryImport:
    repositoryUrl && !register.local && register.sourceType === 'git'
      ? {
          source: repositoryUrl,
          ref: register.ref,
          subpath: register.subpath,
          organization: register.organization || undefined,
          status: register.status,
          workspace: register.workspace,
          trackingUri: register.trackingUri,
          trackingAuth: register.trackingAuth,
        }
      : undefined,
});

export const formatSkillImportCli = ({
  source,
  ref,
  subpath,
  organization,
  status,
  workspace,
  trackingUri,
  trackingAuth,
}: SkillImportSnippetOptions) => {
  const lines = [`mlflow skills import --source ${quoteShellArg(source)}`];
  if (ref) lines.push(`--ref ${quoteShellArg(ref)}`);
  if (subpath) lines.push(`--subpath ${quoteShellArg(subpath)}`);
  if (organization) lines.push(`--organization ${quoteShellArg(organization)}`);
  if (status && status !== SkillStatus.ACTIVE) lines.push(`--status ${status}`);
  return cliCommand(lines, { workspace, trackingUri, trackingAuth });
};

export const formatSkillImportPython = ({
  source,
  ref,
  subpath,
  organization,
  status,
  workspace,
  trackingUri,
  trackingAuth,
}: SkillImportSnippetOptions) => {
  const sourceArgs = [`url=${quotePythonString(source)}`];
  if (ref) sourceArgs.push(`ref=${quotePythonString(ref)}`);
  if (subpath) sourceArgs.push(`subpath=${quotePythonString(subpath)}`);
  const args = [`source=GitSource(${sourceArgs.join(', ')})`];
  if (organization) args.push(`organization=${quotePythonString(organization)}`);
  if (status && status !== SkillStatus.ACTIVE) args.push(`status=${quotePythonString(status)}`);
  return pythonSnippet(
    ['import mlflow', 'from mlflow.genai import GitSource'],
    pythonCall('mlflow.genai.import_skills', args),
    { workspace, trackingUri, trackingAuth },
  );
};

export const formatSkillPullCli = (uri: string, destination: string, connection?: TrackingSnippetConnection) =>
  cliCommand([`mlflow skills pull ${uri}`, `--destination ${destination}`], connection);

export const formatSkillPullPython = (uri: string, destination: string, connection?: TrackingSnippetConnection) =>
  pythonSnippet(
    ['import mlflow.genai'],
    pythonCall('mlflow.genai.pull', [quotePythonString(uri), `destination=${quotePythonString(destination)}`]),
    connection,
  );
