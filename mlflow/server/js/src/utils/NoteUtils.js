import { MLFLOW_INTERNAL_PREFIX } from "./TagUtils";

export const NOTE_TAG_PREFIX = MLFLOW_INTERNAL_PREFIX + 'note.';

export class NoteInfo {
  constructor(content) {
    this.content = content;
  }

  static fromRunTags = (tags) => {
    const obj = Object.values(tags).map((t) =>
      [t.getKey(), t.getValue()]
    ).filter((item) =>
      item[0].startsWith(NOTE_TAG_PREFIX)
    ).reduce((accumulated, item) => {
      const key = item[0].slice(NOTE_TAG_PREFIX.length);
      const newAccumulated = Object.assign({key: item[1]}, accumulated);
      newAccumulated[key] = item[1];
      return newAccumulated;
    }, { });
    return new NoteInfo(obj.content);
  };
}
