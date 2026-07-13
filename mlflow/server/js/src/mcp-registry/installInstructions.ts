import { TransportType } from './types';
import type {
  ServerJSONArgument,
  ServerJSONInput,
  ServerJSONPackage,
  ServerJSONPayload,
  ServerJSONTransport,
} from './types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface OptionalSetting {
  flag: string;
  description: string;
  default?: string;
}

export interface InstructionBlock {
  kind: 'package' | 'remote' | 'fallback';
  label: string;
  claudeCodeCommand: string | null;
  mcpJsonConfig: Record<string, unknown> | null;
  notes: string[];
  fallbackReason?: string;
  fallbackUrl?: string;
  optionalSettings?: OptionalSetting[];
}

export interface InstallInstructions {
  serverName: string;
  blocks: InstructionBlock[];
}

// ---------------------------------------------------------------------------
// Shell utilities
// ---------------------------------------------------------------------------

const SHELL_META = /[;|&`$(){}[\]<>!\n\r\\*?#~]/;

export function hasShellMetacharacters(value: string): boolean {
  return SHELL_META.test(value);
}

export function shellQuote(value: string): string {
  if (value === '') return "''";
  if (!/[^a-zA-Z0-9@_\-+=:.,/]/.test(value)) return value;
  return "'" + value.replace(/'/g, "'\\''") + "'";
}

export function formatMcpJsonSnippet(serverName: string, config: Record<string, unknown>): string {
  return JSON.stringify({ mcpServers: { [serverName]: config } }, null, 2);
}

export function getMcpJsonFooterNote(): string {
  return 'Merge into .mcp.json in your project root (or ~/.claude/.mcp.json for global config), restart your session, and verify with `claude mcp list`.';
}

export function deriveClientName(registryName: string): string {
  const slashIndex = registryName.lastIndexOf('/');
  let raw: string;

  if (slashIndex >= 0) {
    const namespace = registryName.slice(0, slashIndex);
    const slug = registryName.slice(slashIndex + 1);
    const namespaceParts = namespace.split('.');
    const prefix = namespaceParts[namespaceParts.length - 1];
    raw = `${prefix}-${slug}`;
  } else {
    raw = registryName;
  }

  const derived = raw
    .replace(/[^A-Za-z0-9_-]/g, '-')
    .replace(/-{2,}/g, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
  return derived || 'mcp-server';
}

function hasUnsafePackageMetadata(
  identifier: string,
  version: string | undefined,
  registryBaseUrl: string | undefined,
  packageSpec: string,
): boolean {
  return (
    hasShellMetacharacters(identifier) ||
    (version != null && hasShellMetacharacters(version)) ||
    (registryBaseUrl != null && hasShellMetacharacters(registryBaseUrl)) ||
    hasShellMetacharacters(packageSpec)
  );
}

// ---------------------------------------------------------------------------
// Input resolution
// ---------------------------------------------------------------------------

export function substituteTokens(template: string, variables: Record<string, ServerJSONInput>, maxDepth = 10): string {
  if (maxDepth <= 0) return template;
  return template.replace(/\{([^}]+)\}/g, (match, token: string) => {
    const variable = variables[token];
    if (!variable) return match;
    const resolved = resolveInputValue(variable);
    return substituteTokens(resolved, variable.variables ?? {}, maxDepth - 1);
  });
}

function resolveNonSecretValue(input: ServerJSONInput): string {
  if (input.value != null) {
    const vars = input.variables ?? {};
    return Object.keys(vars).length > 0 ? substituteTokens(input.value, vars) : input.value;
  }
  if (input.default != null) return input.default;
  if (input.choices && input.choices.length > 0) {
    return `<${input.choices.join('|')}>`;
  }
  return placeholderFor(input);
}

export function resolveInputValue(input: ServerJSONInput): string {
  if (input.isSecret) return placeholderFor(input);
  return resolveNonSecretValue(input);
}

export function resolveInputValueForJson(input: ServerJSONInput & { name?: string }): string {
  if (input.isSecret) {
    const envName = input.name ?? input.valueHint ?? 'SECRET';
    return `\${${envName.toUpperCase().replace(/[^A-Z0-9_]/g, '_')}}`;
  }
  return resolveNonSecretValue(input);
}

function placeholderFor(input: ServerJSONInput): string {
  if (input.placeholder) return input.placeholder;

  const hint = input.valueHint ?? (input as ServerJSONInput & { name?: string }).name;
  if (hint) {
    if (input.format === 'filepath') return `/path/to/${hint.toLowerCase().replace(/[^a-z0-9]/gi, '-')}`;
    return `<${hint.toLowerCase().replace(/[^a-z0-9_-]/g, '-')}>`;
  }

  if (input.format === 'filepath') return '/path/to/file';
  if (input.format === 'boolean') return 'true';
  if (input.format === 'number') return '0';

  return '<value>';
}

// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------

const KNOWN_RUNNERS = new Set(['npx', 'uvx', 'pipx', 'docker', 'dnx', 'node', 'python', 'bunx']);

const REGISTRY_DEFAULTS = {
  npm: 'npx',
  pypi: 'uvx',
  pip: 'uvx',
  oci: 'docker',
  docker: 'docker',
  nuget: 'dnx',
} satisfies Record<string, string>;

export const RUNNER_SEPARATORS = {
  dnx: '--',
} satisfies Record<string, string>;

const CANONICAL_REGISTRIES = {
  npm: 'https://registry.npmjs.org',
  pypi: 'https://pypi.org/simple',
} satisfies Record<string, string>;

function lookup(map: Record<string, string>, key: string): string | undefined {
  return (map as Record<string, string>)[key];
}

export interface ResolvedRunner {
  runner: string | null;
  prerequisiteNote?: string;
}

export function resolveRunner(runtimeHint: string | undefined, registryType: string): ResolvedRunner {
  if (runtimeHint && KNOWN_RUNNERS.has(runtimeHint)) {
    return { runner: runtimeHint };
  }

  const defaultRunner = lookup(REGISTRY_DEFAULTS, registryType.toLowerCase());

  if (runtimeHint && !KNOWN_RUNNERS.has(runtimeHint)) {
    return {
      runner: defaultRunner ?? null,
      prerequisiteNote: defaultRunner
        ? `Requires ${runtimeHint} (using ${defaultRunner} as runner)`
        : `Requires ${runtimeHint}`,
    };
  }

  return { runner: defaultRunner ?? null };
}

export function buildPackageSpec(identifier: string, version: string | undefined, registryType: string): string {
  if (!version) return identifier;

  const rt = registryType.toLowerCase();
  if (rt === 'npm') return `${identifier}@${version}`;
  if (rt === 'pypi' || rt === 'pip') return `${identifier}==${version}`;

  if (rt === 'oci' || rt === 'docker') {
    if (identifier.includes(':') || identifier.includes('@')) return identifier;
    return `${identifier}:${version}`;
  }

  return `${identifier}@${version}`;
}

export function mapTransportForCli(transport: TransportType): string {
  if (transport === TransportType.STREAMABLE_HTTP) return 'http';
  return transport;
}

export function getRegistryFlag(registryType: string, registryBaseUrl: string | undefined, runner: string): string[] {
  if (!registryBaseUrl) return [];

  const rt = registryType.toLowerCase();
  if (rt === 'oci' || rt === 'docker' || rt === 'mcpb') return [];

  const canonical = lookup(CANONICAL_REGISTRIES, rt);
  if (canonical && registryBaseUrl === canonical) return [];

  if (runner === 'npx') return [`--registry=${registryBaseUrl}`];
  if (runner === 'uvx') return [`--index-url=${registryBaseUrl}`];

  return [];
}

// ---------------------------------------------------------------------------
// Build remote instruction
// ---------------------------------------------------------------------------

function resolveUrl(urlTemplate: string, variables?: Record<string, ServerJSONInput>): string {
  if (!variables || Object.keys(variables).length === 0) return urlTemplate;
  return substituteTokens(urlTemplate, variables);
}

export function buildRemoteInstruction(remote: ServerJSONTransport, serverName: string): InstructionBlock {
  const transport = remote.type;
  const cliTransport = mapTransportForCli(transport);
  const label = `${transport}: ${remote.url ?? '(no URL)'}`;
  const notes: string[] = [];

  if (!remote.url) {
    return {
      kind: 'remote',
      label,
      claudeCodeCommand: null,
      mcpJsonConfig: null,
      notes: [],
      fallbackReason: 'Remote endpoint has no URL defined.',
    };
  }

  const resolvedUrl = resolveUrl(remote.url, remote.variables);
  const headers = remote.headers ?? [];
  const requiredHeaders = headers.filter((h) => h.isRequired !== false);
  const optionalHeaders = headers.filter((h) => h.isRequired === false);

  const cliParts = ['claude mcp add', `--transport ${cliTransport}`, serverName, shellQuote(resolvedUrl)];

  for (const header of requiredHeaders) {
    const value = resolveInputValue(header);
    cliParts.push(`--header ${shellQuote(`${header.name}: ${value}`)}`);
  }

  const mcpConfig = { url: resolvedUrl } as Record<string, unknown>;
  if (transport === TransportType.SSE) {
    mcpConfig['type'] = TransportType.SSE;
  }

  const headerMap: { [key: string]: string } = {};
  for (const header of requiredHeaders) {
    headerMap[header.name] = resolveInputValueForJson(header);
  }
  if (Object.keys(headerMap).length > 0) {
    mcpConfig['headers'] = headerMap;
  }

  if (optionalHeaders.length > 0) {
    notes.push(
      'Optional headers: ' +
        optionalHeaders.map((h) => `${h.name}${h.description ? ` (${h.description})` : ''}`).join(', '),
    );
  }

  return {
    kind: 'remote',
    label,
    claudeCodeCommand: cliParts.join(' '),
    mcpJsonConfig: mcpConfig,
    notes,
    optionalSettings: optionalHeaders.map((h) => ({
      flag: `--header "${h.name}: <value>"`,
      description: h.description ?? h.name,
      default: h.default,
    })),
  };
}

// ---------------------------------------------------------------------------
// Build package instruction
// ---------------------------------------------------------------------------

function isNamedArgument(arg: ServerJSONArgument): arg is ServerJSONArgument & { name: string } {
  return 'name' in arg && typeof (arg as { name?: unknown }).name === 'string';
}

function resolveArgForCli(arg: ServerJSONArgument): string | null {
  if (isNamedArgument(arg)) {
    const value = resolveInputValue(arg);
    if (hasShellMetacharacters(value) && arg.value != null) return null;
    return `${shellQuote(arg.name)} ${shellQuote(value)}`;
  }
  const value = resolveInputValue(arg);
  if (hasShellMetacharacters(value) && arg.value != null) return null;
  return shellQuote(value);
}

function resolveArgForJson(arg: ServerJSONArgument): string[] {
  if (isNamedArgument(arg)) {
    return [arg.name, resolveInputValueForJson(arg)];
  }
  return [resolveInputValueForJson(arg)];
}

export interface FallbackContext {
  websiteUrl?: string;
  repositoryUrl?: string;
}

export function buildPackageInstruction(
  pkg: ServerJSONPackage,
  serverName: string,
  fallbackContext?: FallbackContext,
): InstructionBlock {
  const registryType = pkg.registryType.toLowerCase();
  const label = `${pkg.registryType}: ${pkg.identifier}`;
  const notes: string[] = [];
  const optionalSettings: OptionalSetting[] = [];
  const transport = pkg.transport?.type ?? TransportType.STDIO;

  if (registryType === 'mcpb') {
    return buildMcpbFallback(pkg, label);
  }

  const { runner, prerequisiteNote } = resolveRunner(pkg.runtimeHint, pkg.registryType);
  if (!runner) {
    const fallbackUrl =
      pkg.registryBaseUrl ?? pkg.websiteUrl ?? fallbackContext?.websiteUrl ?? fallbackContext?.repositoryUrl;
    return {
      kind: 'fallback',
      label,
      claudeCodeCommand: null,
      mcpJsonConfig: null,
      notes: prerequisiteNote ? [prerequisiteNote] : [],
      fallbackReason: fallbackUrl
        ? `This package uses a custom registry (${pkg.registryType}). See the publisher's documentation for setup steps.`
        : `This package uses a custom registry (${pkg.registryType}). Contact the publisher for installation instructions.`,
      fallbackUrl,
    };
  }

  if (prerequisiteNote) notes.push(prerequisiteNote);

  const packageSpec = buildPackageSpec(pkg.identifier, pkg.version, pkg.registryType);
  const registryFlags = getRegistryFlag(pkg.registryType, pkg.registryBaseUrl, runner);
  const unsafePackageMetadata = hasUnsafePackageMetadata(pkg.identifier, pkg.version, pkg.registryBaseUrl, packageSpec);
  const runtimeArgs = collectRequiredArgs(pkg.runtimeArguments ?? []);
  const packageArgs = collectRequiredArgs(pkg.packageArguments ?? []);
  const optionalRuntimeArgs = collectOptionalArgs(pkg.runtimeArguments ?? []);
  const optionalPackageArgs = collectOptionalArgs(pkg.packageArguments ?? []);
  const envVars = (pkg.environmentVariables ?? []).filter((e) => e.isRequired !== false);
  const optionalEnvVars = (pkg.environmentVariables ?? []).filter((e) => e.isRequired === false);

  for (const arg of [...optionalRuntimeArgs, ...optionalPackageArgs]) {
    optionalSettings.push({
      flag: isNamedArgument(arg) ? arg.name : (arg.valueHint ?? '<arg>'),
      description: arg.description ?? '',
      default: arg.default,
    });
  }
  for (const env of optionalEnvVars) {
    optionalSettings.push({
      flag: `--env ${env.name}=<value>`,
      description: env.description ?? env.name,
      default: env.default,
    });
  }

  if (transport === TransportType.STDIO) {
    return buildStdioPackage({
      serverName,
      runner,
      packageSpec,
      registryFlags,
      runtimeArgs,
      packageArgs,
      envVars,
      label,
      notes,
      optionalSettings,
      unsafePackageMetadata,
    });
  }

  return buildNetworkPackage({
    serverName,
    runner,
    packageSpec,
    registryFlags,
    runtimeArgs,
    packageArgs,
    envVars,
    transport,
    label,
    notes,
    optionalSettings,
    pkg,
    unsafePackageMetadata,
  });
}

function buildStdioPackage({
  serverName,
  runner,
  packageSpec,
  registryFlags,
  runtimeArgs,
  packageArgs,
  envVars,
  label,
  notes,
  optionalSettings,
  unsafePackageMetadata,
}: {
  serverName: string;
  runner: string;
  packageSpec: string;
  registryFlags: string[];
  runtimeArgs: ServerJSONArgument[];
  packageArgs: ServerJSONArgument[];
  envVars: { name: string; isSecret?: boolean; value?: string; default?: string }[];
  label: string;
  notes: string[];
  optionalSettings: OptionalSetting[];
  unsafePackageMetadata: boolean;
}): InstructionBlock {
  const cliEnvFlags: string[] = [];
  const jsonEnv: { [key: string]: string } = {};
  let unsafeForCli = unsafePackageMetadata;

  for (const env of envVars) {
    const cliValue = resolveInputValue(env);
    const jsonValue = resolveInputValueForJson(env);
    if (env.value != null && hasShellMetacharacters(env.value)) {
      unsafeForCli = true;
    }
    if (runner === 'docker') {
      jsonEnv[env.name] = jsonValue;
    } else {
      cliEnvFlags.push(`--env ${shellQuote(`${env.name}=${cliValue}`)}`);
      jsonEnv[env.name] = jsonValue;
    }
  }

  const cliRuntimeArgs: string[] = [];
  for (const arg of runtimeArgs) {
    const resolved = resolveArgForCli(arg);
    if (resolved === null) {
      unsafeForCli = true;
      break;
    }
    cliRuntimeArgs.push(resolved);
  }

  const cliPackageArgs: string[] = [];
  if (!unsafeForCli) {
    for (const arg of packageArgs) {
      const resolved = resolveArgForCli(arg);
      if (resolved === null) {
        unsafeForCli = true;
        break;
      }
      cliPackageArgs.push(resolved);
    }
  }

  let claudeCodeCommand: string | null = null;
  if (!unsafeForCli) {
    const separator = lookup(RUNNER_SEPARATORS, runner) ?? '';
    const innerParts = [runner, ...registryFlags, ...cliRuntimeArgs];
    if (runner === 'npx') innerParts.push('-y');
    if (runner === 'docker') innerParts.push('run', '-i', '--rm');
    for (const env of envVars) {
      if (runner === 'docker') {
        innerParts.push('-e', shellQuote(env.name));
      }
    }
    innerParts.push(packageSpec);
    if (separator) innerParts.push(separator);
    innerParts.push(...cliPackageArgs);

    const outerParts = ['claude mcp add', `--transport ${TransportType.STDIO}`];
    outerParts.push(...cliEnvFlags);
    outerParts.push(serverName, '--', ...innerParts);
    claudeCodeCommand = outerParts.join(' ');
  }

  const jsonArgs: string[] = [];
  for (const arg of [...registryFlags, ...runtimeArgs.flatMap((a) => resolveArgForJson(a))]) {
    jsonArgs.push(arg);
  }
  if (runner === 'npx') jsonArgs.push('-y');
  if (runner === 'docker') {
    jsonArgs.push('run', '-i', '--rm');
    for (const env of envVars) {
      jsonArgs.push('-e', env.name);
    }
  }
  jsonArgs.push(packageSpec);
  const sep = lookup(RUNNER_SEPARATORS, runner);
  if (sep) jsonArgs.push(sep);
  for (const arg of packageArgs) {
    jsonArgs.push(...resolveArgForJson(arg));
  }

  const mcpJsonConfig: { [key: string]: unknown } = {
    command: runner,
    args: jsonArgs,
  };
  if (Object.keys(jsonEnv).length > 0) {
    mcpJsonConfig['env'] = jsonEnv;
  }

  if (unsafeForCli) {
    notes.push('CLI command omitted due to unsafe characters in fixed values. Use .mcp.json instead.');
  }

  return {
    kind: 'package',
    label,
    claudeCodeCommand,
    mcpJsonConfig,
    notes,
    optionalSettings: optionalSettings.length > 0 ? optionalSettings : undefined,
    fallbackReason: unsafeForCli ? 'Fixed values contain shell metacharacters.' : undefined,
  };
}

function buildNetworkPackage({
  serverName,
  runner,
  packageSpec,
  registryFlags,
  runtimeArgs,
  packageArgs,
  envVars,
  transport,
  label,
  notes,
  optionalSettings,
  pkg,
  unsafePackageMetadata,
}: {
  serverName: string;
  runner: string;
  packageSpec: string;
  registryFlags: string[];
  runtimeArgs: ServerJSONArgument[];
  packageArgs: ServerJSONArgument[];
  envVars: { name: string; isSecret?: boolean; value?: string; default?: string }[];
  transport: TransportType;
  label: string;
  notes: string[];
  optionalSettings: OptionalSetting[];
  pkg: ServerJSONPackage;
  unsafePackageMetadata: boolean;
}): InstructionBlock {
  const cliTransport = mapTransportForCli(transport);
  let unsafeForCli = unsafePackageMetadata;

  const startParts = [runner, ...registryFlags];
  for (const arg of runtimeArgs) {
    const resolved = resolveArgForCli(arg);
    if (resolved === null) {
      unsafeForCli = true;
      break;
    }
    startParts.push(resolved);
  }
  if (!unsafeForCli) {
    if (runner === 'npx') startParts.push('-y');
    startParts.push(packageSpec);
    const sep = lookup(RUNNER_SEPARATORS, runner);
    if (sep) startParts.push(sep);
    for (const arg of packageArgs) {
      const resolved = resolveArgForCli(arg);
      if (resolved === null) {
        unsafeForCli = true;
        break;
      }
      startParts.push(resolved);
    }
  }

  for (const env of envVars) {
    if (env.value != null && hasShellMetacharacters(env.value)) {
      unsafeForCli = true;
      break;
    }
  }

  const envExports = envVars.map((e) => `export ${shellQuote(`${e.name}=${resolveInputValue(e)}`)}`);

  let resolvedUrl = pkg.transport?.url ?? 'http://localhost:3000';
  if (pkg.transport?.url) {
    const allVars: { [key: string]: { value?: string; default?: string } } = {};
    for (const arg of [...(pkg.runtimeArguments ?? []), ...(pkg.packageArguments ?? [])]) {
      if (isNamedArgument(arg)) {
        allVars[arg.name] = arg;
      }
      if (arg.valueHint) {
        allVars[arg.valueHint] = arg;
      }
    }
    for (const env of pkg.environmentVariables ?? []) {
      allVars[env.name] = env;
    }
    resolvedUrl = substituteTokens(resolvedUrl, allVars);
  }

  let claudeCodeCommand: string | null = null;
  if (!unsafeForCli) {
    const step1 = [...envExports, startParts.join(' ')].join('\n');
    const step2 = `claude mcp add --transport ${cliTransport} ${serverName} ${shellQuote(resolvedUrl)}`;
    claudeCodeCommand = `# Step 1: Start the server\n${step1}\n\n# Step 2: Connect\n${step2}`;
  } else {
    notes.push('CLI command omitted due to unsafe characters in fixed values. Use .mcp.json instead.');
  }

  const mcpJsonConfig: { [key: string]: unknown } = {
    url: resolvedUrl,
  };
  if (transport === TransportType.SSE) {
    mcpJsonConfig['type'] = TransportType.SSE;
  }

  notes.push(`This package uses ${transport} transport. Start the server locally (step 1), then connect (step 2).`);
  notes.push('The .mcp.json entry only covers step 2 — the server process from step 1 must be running.');

  return {
    kind: 'package',
    label,
    claudeCodeCommand,
    mcpJsonConfig,
    notes,
    optionalSettings: optionalSettings.length > 0 ? optionalSettings : undefined,
    fallbackReason: unsafeForCli ? 'Fixed values contain shell metacharacters.' : undefined,
  };
}

function buildMcpbFallback(pkg: ServerJSONPackage, label: string): InstructionBlock {
  const notes: string[] = [];
  if (pkg.fileSha256) {
    notes.push(`Verify download: shasum -a 256 <file> should match ${pkg.fileSha256}`);
  } else {
    notes.push('Warning: No fileSha256 provided for this MCPB package.');
  }
  notes.push('Install via a client that supports MCPB bundles (e.g. Claude Desktop extensions).');

  return {
    kind: 'fallback',
    label,
    claudeCodeCommand: null,
    mcpJsonConfig: null,
    notes,
    fallbackReason: 'MCPB packages require a client that supports MCPB bundles.',
    fallbackUrl: pkg.websiteUrl,
  };
}

function collectRequiredArgs(args: ServerJSONArgument[]): ServerJSONArgument[] {
  return args.filter((a) => {
    if (a.isRequired === false) return false;
    if (!isNamedArgument(a) && a.isRequired !== true) return false;
    return true;
  });
}

function collectOptionalArgs(args: ServerJSONArgument[]): ServerJSONArgument[] {
  return args.filter((a) => {
    if (a.isRequired === false) return true;
    if (!isNamedArgument(a) && a.isRequired !== true) return true;
    return false;
  });
}

// ---------------------------------------------------------------------------
// Generate install instructions (orchestrator)
// ---------------------------------------------------------------------------

export function generateInstallInstructions(serverJson: ServerJSONPayload, registryName: string): InstallInstructions {
  const serverName = deriveClientName(registryName);
  const blocks: InstructionBlock[] = [];

  const remotes = serverJson.remotes ?? [];
  const packages = serverJson.packages ?? [];

  for (const remote of remotes) {
    if (remote.type === TransportType.STREAMABLE_HTTP) {
      blocks.push(buildRemoteInstruction(remote, serverName));
    }
  }

  const stdioPackages = packages.filter((p) => (p.transport?.type ?? TransportType.STDIO) === TransportType.STDIO);
  const networkPackages = packages.filter((p) => {
    const t = p.transport?.type ?? TransportType.STDIO;
    return t !== TransportType.STDIO;
  });

  const ctx: FallbackContext = {
    websiteUrl: serverJson.websiteUrl,
    repositoryUrl: serverJson.repository?.url,
  };

  for (const pkg of stdioPackages) {
    blocks.push(buildPackageInstruction(pkg, serverName, ctx));
  }
  for (const pkg of networkPackages) {
    blocks.push(buildPackageInstruction(pkg, serverName, ctx));
  }

  for (const remote of remotes) {
    if (remote.type === TransportType.SSE) {
      blocks.push(buildRemoteInstruction(remote, serverName));
    }
  }

  for (const remote of remotes) {
    if (remote.type !== TransportType.STREAMABLE_HTTP && remote.type !== TransportType.SSE) {
      const fallbackUrl = remote.url ?? serverJson.websiteUrl ?? serverJson.repository?.url;
      blocks.push({
        kind: 'fallback',
        label: `${remote.type}: ${remote.url ?? '(no URL)'}`,
        claudeCodeCommand: null,
        mcpJsonConfig: null,
        notes: [],
        fallbackReason: fallbackUrl
          ? `This endpoint uses ${remote.type} transport, which requires manual setup. See the publisher's documentation for connection steps.`
          : `This endpoint uses ${remote.type} transport, which requires manual setup. Contact the publisher for connection instructions.`,
        fallbackUrl,
      });
    }
  }

  return { serverName, blocks };
}
