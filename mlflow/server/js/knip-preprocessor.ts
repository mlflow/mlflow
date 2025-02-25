import type { Preprocessor } from 'knip';

const preprocess: Preprocessor = (options) => {
  Object.keys(options.issues.exports).forEach((file) => {
    // Ignore unused exports from files starting with "oss_" because they are used in
    // the OSS version. This may overmatch some files that are actually unused in the
    // OSS version but it's hard to check the OSS version in our current setup so it's
    // fine to be conservative.
    if (file.includes('oss_')) {
      // Reduce `exports` counter since the exit code is based on it.
      options.counters.exports -= Object.keys(options.issues.exports[file]).length;
      delete options.issues.exports[file];
      return;
    }
    Object.keys(options.issues.exports[file]).forEach((exportIdentifier) => {
      // Ignore unused exports starting with "oss_" because they are used in the OSS
      // version. See above comment for explanation on why we are being conservative.
      if (exportIdentifier.startsWith('oss_')) {
        // Reduce `exports` counter since the exit code is based on it.
        options.counters.exports -= 1;
        delete options.issues.exports[file][exportIdentifier];
      }
    });
  });
  return options;
};

export default preprocess;
