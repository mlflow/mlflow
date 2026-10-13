module.exports = function pdfjsStripImportMetaLoader(source) {
  this.cacheable();
  return source.replace(/import\.meta\.url/g, '""');
};
