import { takeWhile, truncate } from 'lodash';
// Import pako lazily to reduce bundle size
const lazyPako = () => import('pako');

export const truncateToFirstLineWithMaxLength = (str: string, maxLength: number): string => {
  const truncated = truncate(str, {
    length: maxLength,
  });
  return takeWhile(truncated, (char) => char !== '\n').join('');
};

export const capitalizeFirstChar = (str: unknown) => {
  if (!str || typeof str !== 'string' || str.length < 1) {
    return str;
  }
  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
};

export const middleTruncateStr = (str: string, maxLen: number) => {
  if (str.length > maxLen) {
    const firstPartLen = Math.floor((maxLen - 3) / 2);
    const lastPartLen = maxLen - 3 - firstPartLen;
    return str.substring(0, firstPartLen) + '...' + str.substring(str.length - lastPartLen, str.length);
  } else {
    return str;
  }
};

const capitalizeFirstLetter = (string: string) => {
  return string.charAt(0).toUpperCase() + string.slice(1);
};

const _keyStr = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=';

/* eslint-disable no-bitwise */
/**
 * UTF-8 safe version of base64 encoder
 * Source: http://www.webtoolkit.info/javascript_base64.html
 *
 * @param {string} input - Text to encode
 */
export const btoaUtf8 = (input: string) => {
  let output = '';
  let i = 0;

  const result = _utf8_encode(input);

  while (i < result.length) {
    const chr1 = result.charCodeAt(i++);
    const chr2 = result.charCodeAt(i++);
    const chr3 = result.charCodeAt(i++);

    const enc1 = chr1 >> 2;
    const enc2 = ((chr1 & 3) << 4) | (chr2 >> 4);
    let enc3 = ((chr2 & 15) << 2) | (chr3 >> 6);
    let enc4 = chr3 & 63;

    if (isNaN(chr2)) {
      enc4 = 64;
      enc3 = enc4;
    } else if (isNaN(chr3)) {
      enc4 = 64;
    }

    output = output + _keyStr.charAt(enc1) + _keyStr.charAt(enc2) + _keyStr.charAt(enc3) + _keyStr.charAt(enc4);
  }

  return output;
};

/**
 * UTF-8 safe version of base64 decoder
 * Source: http://www.webtoolkit.info/javascript_base64.html
 *
 * @param {string} input - Text to decode
 */
export const atobUtf8 = (input: string) => {
  let output = '';
  let i = 0;

  const result = input?.replace(/[^A-Za-z0-9+/=]/g, '') || '';

  while (i < result.length) {
    const enc1 = _keyStr.indexOf(result.charAt(i++));
    const enc2 = _keyStr.indexOf(result.charAt(i++));
    const enc3 = _keyStr.indexOf(result.charAt(i++));
    const enc4 = _keyStr.indexOf(result.charAt(i++));

    const chr1 = (enc1 << 2) | (enc2 >> 4);
    const chr2 = ((enc2 & 15) << 4) | (enc3 >> 2);
    const chr3 = ((enc3 & 3) << 6) | enc4;

    output += String.fromCharCode(chr1);

    if (enc3 !== 64) {
      output += String.fromCharCode(chr2);
    }

    if (enc4 !== 64) {
      output += String.fromCharCode(chr3);
    }
  }

  return _utf8_decode(output);
};

/**
 * (private method) does a UTF-8 encoding
 *
 * @private
 * @param {string} string - Text to encode
 */
const _utf8_encode = (string = '') => {
  const result = string.replace(/\r\n/g, '\n');
  let utftext = '';

  for (let n = 0; n < result.length; n++) {
    const c = result.charCodeAt(n);

    if (c < 128) {
      utftext += String.fromCharCode(c);
    } else if (c > 127 && c < 2048) {
      utftext += String.fromCharCode((c >> 6) | 192) + String.fromCharCode((c & 63) | 128);
    } else {
      utftext +=
        String.fromCharCode((c >> 12) | 224) +
        String.fromCharCode(((c >> 6) & 63) | 128) +
        String.fromCharCode((c & 63) | 128);
    }
  }

  return utftext;
};

/**
 * (private method) does a UTF-8 decoding
 *
 * @private
 * @param {string} utftext - UTF-8 text to dencode
 */
const _utf8_decode = (utftext = '') => {
  let string = '';
  let i = 0;

  while (i < utftext.length) {
    const c = utftext.charCodeAt(i);

    if (c < 128) {
      string += String.fromCharCode(c);
      i++;
    } else if (c > 191 && c < 224) {
      const c2 = utftext.charCodeAt(i + 1);
      string += String.fromCharCode(((c & 31) << 6) | (c2 & 63));
      i += 2;
    } else {
      const c2 = utftext.charCodeAt(i + 1);
      const c3 = utftext.charCodeAt(i + 2);
      string += String.fromCharCode(((c & 15) << 12) | ((c2 & 63) << 6) | (c3 & 63));
      i += 3;
    }
  }
  return string;
};
/* eslint-enable no-bitwise */

/**
 * Returns a SHA256 hash of the input string
 */
export const getStringSHA256 = (input: string) => {
  return crypto.subtle.digest('SHA-256', new TextEncoder().encode(input)).then((arrayBuffer) => {
    return Array.prototype.map.call(new Uint8Array(arrayBuffer), (x) => ('00' + x.toString(16)).slice(-2)).join('');
  });
};

const COMPRESSED_TEXT_DEFLATE_PREFIX = 'deflate;';

export const textCompressDeflate = async (text: string) => {
  const pako = await lazyPako();
  const binaryData = pako.deflate(text);

  // Buffer-based implementation
  if (typeof Buffer !== 'undefined') {
    const b64encoded = Buffer.from(binaryData).toString('base64');
    return `${COMPRESSED_TEXT_DEFLATE_PREFIX}${b64encoded}`;
  }

  // btoa-based implementation
  const binaryString = Array.from(binaryData, (byte) => String.fromCodePoint(byte)).join('');
  return `${COMPRESSED_TEXT_DEFLATE_PREFIX}${btoa(binaryString)}`;
};

export const textDecompressDeflate = async (compressedText: string) => {
  const pako = await lazyPako();
  if (!compressedText.startsWith(COMPRESSED_TEXT_DEFLATE_PREFIX)) {
    throw new Error('Invalid compressed text, payload header invalid');
  }
  const compressedTextWithoutPrefix = compressedText.slice(COMPRESSED_TEXT_DEFLATE_PREFIX.length);

  // Buffer-based implementation
  if (typeof Buffer !== 'undefined') {
    const binaryString = Buffer.from(compressedTextWithoutPrefix, 'base64');
    return pako.inflate(
      // This doesn't fail in Mlflow-Copybara-Tester-Pr. TODO: check why.
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore [FEINF-4084] No overload matches this call.
      binaryString,
      { to: 'string' },
    );
  }

  // atob-based implementation
  const binaryString = atob(compressedTextWithoutPrefix);
  return pako.inflate(
    Uint8Array.from(binaryString, (m) => m.codePointAt(0) ?? 0),
    { to: 'string' },
  );
};

export const isTextCompressedDeflate = (text: string) => text.startsWith(COMPRESSED_TEXT_DEFLATE_PREFIX);
