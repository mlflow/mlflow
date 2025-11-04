const OPTIONAL_INTEGRATIONS = ['mlflow-openai', 'mlflow-anthropic', 'mlflow-vercel'];

/**
 * Attempt to load optional integration packages when they are installed.
 * Any missing module is silently ignored. Other failures are logged.
 */
export function tryEnableOptionalIntegrations(): void {
  for (const integration of OPTIONAL_INTEGRATIONS) {
    importOptionalModule(integration);
  }
}

function importOptionalModule(moduleId: string): void {
  try {
    require(moduleId);
  } catch (error) {
    if (isModuleNotFound(error, moduleId)) {
      return;
    }

    console.warn(`Failed to load optional integration '${moduleId}':`, error);
  }
}

function isModuleNotFound(error: unknown, moduleId: string): boolean {
  if (!error || typeof error !== 'object') {
    return false;
  }

  const err = error as { code?: string; message?: string };
  const isModuleMissing = err.code === 'MODULE_NOT_FOUND';
  const mentionsModule = typeof err.message === 'string' ? err.message.includes(moduleId) : false;
  return Boolean(isModuleMissing && mentionsModule);
}
