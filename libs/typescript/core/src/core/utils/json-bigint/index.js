/**
 * NOTE: The contents of this file have been inlined from the json-bigint package's source code
 * https://github.com/sidorares/json-bigint/blob/master/json-bigint.js
 *
 * The repository contains a critical bug fix for decimal handling, however, it has not been
 * released to npm yet. This file is a copy of the source code with the bug fix applied.
 * https://github.com/sidorares/json-bigint/commit/3530541b016d9041db6c1e7019e6999790bfd857
 *
 * :copyright: Copyright (c) 2013 Andrey Sidorov
 * :license: The MIT License (MIT)
 */

// @ts-nocheck
var json_stringify = require('./stringify.js').stringify;
var json_parse = require('./parse.js');

module.exports = function (options) {
  return {
    parse: json_parse(options),
    stringify: json_stringify
  };
};
//create the default method members with no options applied for backwards compatibility
module.exports.parse = json_parse();
module.exports.stringify = json_stringify;
