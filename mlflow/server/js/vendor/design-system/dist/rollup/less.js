import path from 'node:path';
// eslint-disable-next-line import/no-extraneous-dependencies
import { less } from '@databricks/pkgr';
class CustomFileManager extends less.FileManager {
    async loadFile(filename, currentDirectory, options, environment) {
        // Make `@import './rtl'` a no-op. These imports exist in antd .less files and consistently
        // referred as './rtl'. This reduces bundle size.
        // While we could use PostcssFilterRules to filter out -rtl after they are bundled, this approach
        // speeds the build up.
        if (filename === './rtl') {
            filename = path.join(__dirname, 'noop.less');
        }
        // Support `@import '~antd/xxx'` syntax and direct it to node_modules
        else if (filename.startsWith('~')) {
            filename = require.resolve(filename.substring(1));
        }
        return super.loadFile(filename, currentDirectory, options, environment);
    }
}
export const lessCustomFileManager = {
    install(_lessInstance, pluginManager) {
        pluginManager.addFileManager(new CustomFileManager());
    },
};
//# sourceMappingURL=less.js.map