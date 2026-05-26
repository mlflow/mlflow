import type { MessageDescriptor } from 'react-intl';
import { defineMessage } from 'react-intl';

/**
 * Git source tags MLflow records on runs via `GitRunContext`, plus the friendly labels we use
 * for them in two UI contexts:
 *   - short:  used in dropdowns / selectors that sit under a "Git" section header
 *             (e.g. "commit", "branch", "repository")
 *   - column: used as the table column header and grouped-run row label
 *             (e.g. "Git commit", "Git branch", "Git repository")
 */
type GitSourceTagLabels = {
  short: MessageDescriptor;
  column: MessageDescriptor;
};

export const GIT_SOURCE_TAGS = {
  'mlflow.source.git.commit': {
    short: defineMessage({
      defaultMessage: 'commit',
      description: 'Short label for the git commit tag in MLflow UI dropdowns and selectors',
    }),
    column: defineMessage({
      defaultMessage: 'Git commit',
      description: 'Column header / row label for the git commit tag in MLflow tables',
    }),
  },
  'mlflow.source.git.branch': {
    short: defineMessage({
      defaultMessage: 'branch',
      description: 'Short label for the git branch tag in MLflow UI dropdowns and selectors',
    }),
    column: defineMessage({
      defaultMessage: 'Git branch',
      description: 'Column header / row label for the git branch tag in MLflow tables',
    }),
  },
  'mlflow.source.git.repoURL': {
    short: defineMessage({
      defaultMessage: 'repository',
      description: 'Short label for the git repository tag in MLflow UI dropdowns and selectors',
    }),
    column: defineMessage({
      defaultMessage: 'Git repository',
      description: 'Column header / row label for the git repository tag in MLflow tables',
    }),
  },
} satisfies Record<string, GitSourceTagLabels>;

export type GitSourceTagKey = keyof typeof GIT_SOURCE_TAGS;

export const GIT_SOURCE_TAG_KEYS: Set<string> = new Set(Object.keys(GIT_SOURCE_TAGS));

export const isGitSourceTag = (tag: string): tag is GitSourceTagKey => tag in GIT_SOURCE_TAGS;
