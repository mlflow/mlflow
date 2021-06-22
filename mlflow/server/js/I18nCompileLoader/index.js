const loaderUtils = require('loader-utils');
const { resolveBuiltinFormatter } = require('@formatjs/cli/src/formatters');

/**
 * This is an async loader we use to compile localization resources and write back.
 *   Compare to FormatJS native compile function, it only support `format` option.
 * @param content resource file content
 */
module.exports = async function i18nLoader(content) {
  this.cacheable();
  const callback = this.async();
  const { format } = loaderUtils.getOptions(this) || {};
  const formatter = await resolveBuiltinFormatter(format);
  const resourceFileContent = JSON.parse(content);
  const compiledContent = await formatter.compile(resourceFileContent);
  return callback(null, JSON.stringify(compiledContent));
};
