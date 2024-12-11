import fs from 'node:fs';
import path from 'node:path';

import { expect } from '@jest/globals';
import { minimatch } from 'minimatch';

const { testPath } = expect.getState();

const userDir = process.cwd();

const relativeTestPath = testPath && path.relative(userDir, testPath);

const JEST_REACT_VERSIONS_FILE = '.jestreactversions';

const userJestReact17FilePath = path.join(userDir, JEST_REACT_VERSIONS_FILE);

interface JestReactVersionModifiers {
  reactVersion: 17 | 18;
  rtlVersion: 12 | 14;
}

/**
 * Converts lines in .jestreactversions to JestReactVersionEntry
 * @param line a csv string formatted as `pattern,react:18,rtl:14`
 */
function convertLineToEntry(line: string) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_pattern, ...rest] = line.split(/,/);

  // Defaults
  let reactVersion: JestReactVersionModifiers['reactVersion'] = DEFAULT_MODIFIERS_FOR_LISTED_FILES.reactVersion;
  let rtlVersion: JestReactVersionModifiers['rtlVersion'] = DEFAULT_MODIFIERS_FOR_LISTED_FILES.rtlVersion;

  for (const modifier of rest) {
    if (modifier === 'react:18') {
      reactVersion = 18;
    }
    if (modifier === 'rtl:14') {
      rtlVersion = 14;
    }
  }

  if (reactVersion === 17 && rtlVersion === 14) {
    throw new Error('RTL 14 is not compatible with React 17');
  }

  const entry: JestReactVersionModifiers = {
    reactVersion,
    rtlVersion,
  };

  return entry;
}

function getPatternFromLine(line: string): string {
  return line.split(',')[0];
}

// TODO: Clean this up, .jestreactversions file does not exist anymore
const userJestReactVersionsLines = [];

// Check if the current file fits any of the patterns in the .jestreactversions file
const currentTestLine = relativeTestPath
  ? userJestReactVersionsLines.find((line) => minimatch(relativeTestPath, getPatternFromLine(line)))
  : undefined;

// Files that are listed with no modifiers get these defaults
const DEFAULT_MODIFIERS_FOR_LISTED_FILES: JestReactVersionModifiers = {
  reactVersion: 17,
  rtlVersion: 12,
};

// Represents the latest versions and APIs to future proof new tests and existing tests that support this combination.
const DEFAULT_MODIFIERS_FOR_UNLISTED_FILES: JestReactVersionModifiers = {
  reactVersion: 18,
  rtlVersion: 14,
};

const currentEntry = currentTestLine ? convertLineToEntry(currentTestLine) : DEFAULT_MODIFIERS_FOR_UNLISTED_FILES;

export { currentEntry };
