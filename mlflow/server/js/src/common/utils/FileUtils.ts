export const getBasename = (path: string) => {
  const parts = path.split('/');
  return parts[parts.length - 1];
};

export const getExtension = (path: string) => {
  const parts = path.split(/[./]/);
  return parts[parts.length - 1];
};

export const getLanguage = (path: string) => {
  const ext = getExtension(path).toLowerCase();
  if (ext in MLFLOW_FILE_LANGUAGES) {
    return MLFLOW_FILE_LANGUAGES[ext];
  }
  return ext;
};

const MLPROJECT_FILE_NAME = 'mlproject';
const MLMODEL_FILE_NAME = 'mlmodel';

const MLFLOW_FILE_LANGUAGES = {
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
  // Common code extensions
  'c',
  'cpp',
  'h',
  'hpp',
  'java',
  'sh',
  'bash',
  'r',
  'scala',
  'sql',
  'go',
  'rb',
  'php',
  'swift',
  'rust',
  'kt',
  'kts',
]);
export const HTML_EXTENSIONS = new Set(['html']);
export const MAP_EXTENSIONS = new Set(['geojson']);
export const PDF_EXTENSIONS = new Set(['pdf']);
export const DATA_EXTENSIONS = new Set(['csv', 'tsv']);
// Audio extensions supported by wavesurfer.js
// Source https://github.com/katspaugh/wavesurfer.js/discussions/2703#discussioncomment-5259526
export const AUDIO_EXTENSIONS = new Set(['m4a', 'mp3', 'mp4', 'wav', 'aac', 'wma', 'flac', 'opus', 'ogg']);

export const isTextFile = (path: string, content?: string, size?: number): boolean => {
  const ext = getExtension(path).toLowerCase();

  if (TEXT_EXTENSIONS.has(ext)) {
    return true;
  }

  if (content) {
    // Check for null bytes which typically indicate binary data
    if (content.indexOf('\0') >= 0) {
      return false;
    }

    // If the content has reasonable line breaks and printable characters, it's likely text
    const hasReasonableLines = content.split('\n').length > 1;
    const isPrintable = /^[\x20-\x7E\t\n\r\s]*$/.test(content.slice(0, 1000));

    return hasReasonableLines && isPrintable;
  }

  // If content is not available, make a conservative guess based on non-classified extensions
  // Avoid files that are likely binary but don't have known extensions
  const knownBinaryExts = new Set([
    ...Array.from(IMAGE_EXTENSIONS),
    ...Array.from(AUDIO_EXTENSIONS),
    ...Array.from(PDF_EXTENSIONS),
    'exe', 'dll', 'so', 'dylib', 'bin', 'dat', 'db', 'sqlite', 'zip', 'tar', 'gz', 'rar', '7z'
  ]);

  if (knownBinaryExts.has(ext)) {
    return false;
  }

  // Default to text for small files with unknown extensions
  const MAX_DEFAULT_TEXT_SIZE = 1024 * 1024; // 1MB
  return !size || size < MAX_DEFAULT_TEXT_SIZE;
};
