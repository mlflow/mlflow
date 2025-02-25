import path from 'node:path';

// eslint-disable-next-line import/no-extraneous-dependencies
import { less } from '@databricks/pkgr';

const getPnpApi = async () => {
  try {
    // yarn itself injects pnpapi import if the process is running with pnp
    return await import('pnpapi');
  } catch {
    // not in pnp, no need to use pnp api
  }
};

const getResolvedPath = async (importee: string, importer: string): Promise<string> => {
  const pnpapi = await getPnpApi();
  const resolvedPath = pnpapi ? pnpapi.resolveToUnqualified(importee, importer) : require.resolve(importee);
  return resolvedPath as string;
};

class CustomFileManager extends less.FileManager implements Less.FileManager {
  async loadFile(
    filename: string,
    currentDirectory: string,
    options: Less.LoadFileOptions,
    environment: Less.Environment,
  ): Promise<Less.FileLoadResult> {
    // Make `@import './rtl'` a no-op. These imports exist in antd .less files and consistently
    // referred as './rtl'. This reduces bundle size.
    // While we could use PostcssFilterRules to filter out -rtl after they are bundled, this approach
    // speeds the build up.
    if (filename === './rtl') {
      filename = path.join(__dirname, 'noop.less');
    }
    // Support `@import '~antd/xxx'` syntax and direct it to node_modules using yarn pnp api
    else if (filename.startsWith('~')) {
      filename = await getResolvedPath(filename.substring(1), currentDirectory);
    }

    return super.loadFile(filename, currentDirectory, options, environment);
  }
}

export const lessCustomFileManager: Less.Plugin = {
  install(_lessInstance, pluginManager) {
    pluginManager.addFileManager(new CustomFileManager());
  },
};
