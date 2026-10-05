// Copyable CLI and Python examples. Every user-supplied value goes through a quoting helper.
import { SkillStatus } from './types';
import type { SkillRegistrationSourceType } from './sourceLocation';

const quoteShellArg = (value: string) => `'${value.replace(/'/g, "'\\''")}'`;

// JSON string escapes (\", \\, \n, \uXXXX) are all valid Python string-literal escapes.
const quotePythonString = (value: string) => JSON.stringify(value);

const cliCommand = (lines: string[]) => lines.map((line, index) => (index === 0 ? line : `    ${line}`)).join(' \\\n');

const pythonCall = (call: string, args: string[]) => `${call}(\n${args.map((arg) => `    ${arg},`).join('\n')}\n)`;

const PYTHON_SOURCE_CLASS: Record<SkillRegistrationSourceType, string> = {
  git: 'GitSource',
  oci: 'OCISource',
  zip: 'ZipSource',
};

export interface SkillRegisterSnippetOptions {
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
}: SkillRegisterSnippetOptions) => {
  const lines = [
    local ? `mlflow skills register ${quoteShellArg('<directory>')}` : `mlflow skills register ${sourceType}`,
  ];
  if (name) lines.push(`--name ${quoteShellArg(name)}`);
  if (organization) lines.push(`--organization ${quoteShellArg(organization)}`);
  if (!local) {
    lines.push(`${sourceType === 'oci' ? '--image' : '--url'} ${quoteShellArg(location.trim() || '<location>')}`);
    if (ref && sourceType === 'git') lines.push(`--ref ${quoteShellArg(ref)}`);
    if (subpath) lines.push(`--subpath ${quoteShellArg(subpath)}`);
  }
  if (status && status !== SkillStatus.ACTIVE) lines.push(`--status ${status}`);
  return cliCommand(lines);
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
  return [...imports, '', pythonCall('mlflow.genai.register_skill', args)].join('\n');
};

export interface SkillImportSnippetOptions {
  source: string;
  ref?: string;
  /** Discovery root; the import registers every SKILL.md found beneath it. */
  subpath?: string;
  organization?: string;
}

export const formatSkillImportCli = ({ source, ref, subpath, organization }: SkillImportSnippetOptions) => {
  const lines = [`mlflow skills import --source ${quoteShellArg(source)}`];
  if (ref) lines.push(`--ref ${quoteShellArg(ref)}`);
  if (subpath) lines.push(`--subpath ${quoteShellArg(subpath)}`);
  if (organization) lines.push(`--organization ${quoteShellArg(organization)}`);
  return cliCommand(lines);
};

export const formatSkillImportPython = ({ source, ref, subpath, organization }: SkillImportSnippetOptions) => {
  const sourceArgs = [`url=${quotePythonString(source)}`];
  if (ref) sourceArgs.push(`ref=${quotePythonString(ref)}`);
  if (subpath) sourceArgs.push(`subpath=${quotePythonString(subpath)}`);
  const typed = Boolean(ref || subpath);
  const args = [typed ? `source=GitSource(${sourceArgs.join(', ')})` : `source=${quotePythonString(source)}`];
  if (organization) args.push(`organization=${quotePythonString(organization)}`);
  const imports = typed ? ['import mlflow', 'from mlflow.genai import GitSource'] : ['import mlflow'];
  return [...imports, '', pythonCall('mlflow.genai.import_skills', args)].join('\n');
};

export const formatSkillPullCli = (uri: string, destination: string) =>
  cliCommand([`mlflow skills pull ${uri}`, `--destination ${destination}`]);

export const formatSkillPullPython = ({
  name,
  organization,
  version,
  destination,
}: {
  name: string;
  organization?: string;
  version?: number;
  destination: string;
}) => {
  const args = [`name=${quotePythonString(name)}`];
  if (organization) args.push(`organization=${quotePythonString(organization)}`);
  if (version != null) args.push(`version=${version}`);
  args.push(`destination=${quotePythonString(destination)}`);
  return ['import mlflow.genai', '', pythonCall('mlflow.genai.pull', args)].join('\n');
};
