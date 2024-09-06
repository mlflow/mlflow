/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

export const getBasename = (path: any) => {
  const parts = path.split('/');
  return parts[parts.length - 1];
};

export const getExtension = (path: any) => {
  const parts = path.split(/[./]/);
  return parts[parts.length - 1];
};

export const getLanguage = (path: any) => {
  const ext = getExtension(path).toLowerCase();
  if (ext in MLFLOW_FILE_LANGUAGES) {
    return MLFLOW_FILE_LANGUAGES[ext];
  }
  return ext;
};

export const MLPROJECT_FILE_NAME = 'mlproject';
export const MLMODEL_FILE_NAME = 'mlmodel';

export const MLFLOW_FILE_LANGUAGES = {
  [MLPROJECT_FILE_NAME.toLowerCase()]: 'yaml',
  [MLMODEL_FILE_NAME.toLowerCase()]: 'yaml',
};

export const IMAGE_EXTENSIONS = new Set(['jpg', 'bmp', 'jpeg', 'png', 'gif', 'svg']);
export const TEXT_EXTENSIONS = new Set([
  'txt',
  'log',
  'err',
  'cfg',
  'conf',
  'cnf',
  'cf',
  'ini',
  'properties',
  'prop',
  'hocon',
  'toml',
  'yaml',
  'yml',
  'xml',
  'json',
  'js',
  'py',
  'py3',
  'md',
  'rst',
  MLPROJECT_FILE_NAME.toLowerCase(),
  MLMODEL_FILE_NAME.toLowerCase(),
  'jsonnet',
]);
export const HTML_EXTENSIONS = new Set(['html']);
export const MAP_EXTENSIONS = new Set(['geojson']);
export const PDF_EXTENSIONS = new Set(['pdf']);
export const DATA_EXTENSIONS = new Set(['csv', 'tsv']);
// Audio extensions supported by wavesurfer.js
// Source https://github.com/katspaugh/wavesurfer.js/discussions/2703#discussioncomment-5259526
export const AUDIO_EXTENSIONS = new Set(['m4a', 'mp3', 'mp4', 'wav', 'aac', 'wma', 'flac', 'opus', 'ogg']);
