# Tasks

## Rules
- You must ask the dev for approval whenever you finish a task and before you start a new task.
- Before marking a task as done, you must check it via playwright MCP tool.

## Checklist
- [x] Secure UX sign-off for the home page mock; confirm design system icons will be reused and note placeholder thumbnail locations in `/docs`.
- [ ] Update routing/navigation to add `RoutePaths.home`, register `PageId.home`, wire sidebar and logo link to `/`.
- [ ] Build `HomePage.tsx` with hero, quick-action cards, experiments table, news grid.
- [ ] Reuse experiment data slice for latest experiments, load static news data with error handling.
