/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { MLFLOW_LOGGED_ARTIFACTS_TAG } from '../constants';
import { RunLoggedArtifactType, type RunLoggedArtifactsDeclaration } from '../types';
import type { KeyValueEntity } from '../../common/types';

export class ArtifactNode {
  children: any;
  fileInfo: any;
  isLoaded: any;
  isRoot: any;
  constructor(isRoot: any, fileInfo: any, children: any) {
    this.isRoot = isRoot;
    this.isLoaded = false;
    // fileInfo should not be defined for the root node.
    this.fileInfo = fileInfo;
    // map of basename to ArtifactNode
    this.children = children;
  }

  deepCopy() {
    const node = new ArtifactNode(this.isRoot, this.fileInfo, undefined);
    node.isLoaded = this.isLoaded;
    if (this.children) {
      const copiedChildren = {};
      Object.keys(this.children).forEach((name) => {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        copiedChildren[name] = this.children[name].deepCopy();
      });
      node.children = copiedChildren;
    }
    return node;
  }

  setChildren(fileInfos: any) {
    if (fileInfos) {
      this.children = {};
      this.isLoaded = true;
      fileInfos.forEach((fileInfo: any) => {
        // basename is the last part of the path for this fileInfo.
        const pathParts = fileInfo.path.split('/');
        const basename = pathParts[pathParts.length - 1];
        let children;
        if (fileInfo.is_dir) {
          children = [];
        }
        this.children[basename] = new ArtifactNode(false, fileInfo, children);
      });
    } else {
      this.isLoaded = true;
    }
  }

  static findChild(node: any, path: any) {
    // Filter out empty strings caused by spurious instances of slash, i.e.
    // "model/" instead of just "model"
    const parts = path.split('/').filter((item: any) => item);
    let ret = node;
    parts.forEach((part: any) => {
      if (ret.children && ret.children[part] !== undefined) {
        ret = ret.children[part];
      } else {
        throw new Error("Can't find child.");
      }
    });
    return ret;
  }

  static isEmpty(node: any) {
    return node.children === undefined || Object.keys(node.children).length === 0;
  }
}

/**
 * Extracts the list of tables logged in the run from the run tags.
 */
export const extractLoggedTablesFromRunTags = (runTags: Record<string, KeyValueEntity>) => {
  const rawLoggedArtifactsDeclaration = runTags?.[MLFLOW_LOGGED_ARTIFACTS_TAG]?.value;
  const tablesInRun: Set<string> = new Set();
  if (rawLoggedArtifactsDeclaration) {
    try {
      const loggedArtifacts: RunLoggedArtifactsDeclaration = JSON.parse(rawLoggedArtifactsDeclaration);

      loggedArtifacts
        .filter(({ type }) => type === RunLoggedArtifactType.TABLE)
        .forEach(({ path }) => {
          tablesInRun.add(path);
        });
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new SyntaxError(`The "${MLFLOW_LOGGED_ARTIFACTS_TAG}" tag is malformed!`);
      }
      throw error;
    }
  }
  return Array.from(tablesInRun);
};
