import { MLFLOW_INTERNAL_PREFIX } from "./TagUtils";

export const NOTE_TAG_PREFIX = MLFLOW_INTERNAL_PREFIX + 'note.';

export class NoteInfo {
  constructor(content) {
    this.content = content;
  }

  static fromRunTags = (tags) => {
    const contentTag = Object.values(tags).find((t) => t.getKey() === 'mlflow.note.content');
    if (contentTag === undefined) {
      return undefined;
    }
    return new NoteInfo(contentTag.getValue());
  };
}
