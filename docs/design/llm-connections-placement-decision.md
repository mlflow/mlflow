# LLM Connections placement decision

**Decision: Option A — revert to a dedicated LLM Connections tab/page under Settings.**

Restore `llm-connections` as a first-class settings section (sidebar item + path
segment + standalone render), drop the General embedding, and turn the two redirect
surfaces into aliases that point at the standalone route again. This branch's IA
commit is cleanly self-contained, so the revert blast radius is small.

## Rationale

- **The altitude mismatch is real and structural, not cosmetic.** `ApiKeysPageInner`
  is a full page-weight CRUD experience — a filterable/searchable multi-column table
  (Key name / Provider / Allowed models / Endpoints / Used by / Last updated) with
  Add/Delete toolbar and four drawers/modals. General's other rows are single-line
  toggle/button cards (`SettingsRow`). Embedding a table-plus-toolbar under a list of
  toggles is what reads as "weird." Restyling it into a compact summary card (Option B)
  would require building a *second* surface for the same manager and a click-out — more
  code, more drift, and it hides the very thing the user wanted front-and-center.

- **Precedent points the other way.** Webhooks is the closest analog in this repo: it
  is *also* a full CRUD table (list + Create button + form/delete modals in
  `WebhooksSettings.tsx`, 292 lines) and it lives on its **own** dedicated tab
  `/settings/webhooks`. Before this branch, LLM Connections was consistent with
  Webhooks. This branch made LLM Connections the *inconsistent* one — the only heavy
  CRUD surface crammed into a shared section. Reverting restores parity ("one heavy
  manager = one tab").

- **A dedicated PAGE satisfies "front and center" better than a General section does.**
  The user's original goal was discoverability. But a section buried *below* Theme /
  Telemetry / Demo-data cards in General (reachable only by scrolling, or via a hash
  that smooth-scrolls) is arguably *less* discoverable than a top-level, always-visible
  sidebar item sitting right next to General and Webhooks. The earlier "buried tab"
  complaint is resolved not by merging into General, but by keeping it a sibling tab in
  the same sub-sidebar the user already sees. That reconciles the tension: promote it to
  a peer tab, don't demote it to a sub-section.

- **The revert is clean.** The whole IA move is 4 commits, and the placement change is
  concentrated in ~4 files (`SettingsPage.tsx`, `settingsSectionConstants.ts`,
  `MlflowSidebarSettingsItems.tsx`, `route-defs.ts`) plus 2 link/redirect touch-points.
  The valuable, unrelated work on this branch — **multiple models per connection /
  allowlist** (`ModelAllowlistField`, `useAllowlistedModelPairs`, `ModelSelectorModal`,
  `CreateApiKeyModal`, `gateway/api.ts`, DB migration) — is orthogonal to placement and
  MUST be kept. Only the placement/IA hunk is reverted.

## Concrete change list

Keep everything under `gateway/components/model-selector/`,
`hooks/useAllowlistedModelPairs*`, `ApiKeysList`, `CreateApiKeyModal`, `EditApiKeyModal`,
`gateway/api.ts`, `types.ts`, and the allowlist migration. Only reverse the IA:

1. `settings/settingsSectionConstants.ts`
   - Add `SETTINGS_SECTION_LLM_CONNECTIONS` back into `SETTINGS_PATH_SEGMENTS`
     (`[GENERAL, LLM_CONNECTIONS, WEBHOOKS]`) and restore the plain doc comment
     (drop the "legacy / intentionally absent" wording).

2. `common/components/MlflowSidebarSettingsItems.tsx`
   - Re-import `SETTINGS_SECTION_LLM_CONNECTIONS` and re-add the `MlflowSidebarLink`
     for it (componentId `mlflow.sidebar.settings_llm_connections_link`), placed between
     the General and Webhooks links.

3. `settings/SettingsPage.tsx`
   - Remove the `useEffect` redirect for unknown segments (lines ~94-107) and the
     hash-anchor scroll `useEffect` (~109-122) plus the `llmConnectionsRef`.
   - Move the `<ApiKeysPageInner />` block out of the `general` branch into its own
     `activeSection === 'llm-connections'` branch, mirroring the Webhooks branch
     (a `SettingsSectionHeader` + `<ApiKeysPageInner />`). General reverts to just the
     preferences card.
   - Drop the now-unused `useLocation`/`useNavigate`/`SETTINGS_RETURN_TO_PARAM` imports
     if no longer referenced.

4. `experiment-tracking/route-defs.ts`
   - Restore the `case 'llm-connections': return 'Settings – LLM Connections';`
     page-title arm.

5. Deep links / redirect targets — point back at the standalone route:
   - `gateway/components/edit-endpoint/EditEndpointFormRenderer.tsx`: change the link
     back to `Routes.getSettingsSectionRoute(SETTINGS_SECTION_LLM_CONNECTIONS)`
     (drop the `#llm-connections` hash form), and drop the now-unused
     `SETTINGS_SECTION_GENERAL` import.
   - `experiment-tracking/.../traces-v3/GenAIModelSelection.tsx`: change
     `CONNECTIONS_ROUTE` back to `getSettingsSectionRoute(SETTINGS_SECTION_LLM_CONNECTIONS)`
     (no hash).
   - `gateway/pages/RedirectApiKeysToSettings.tsx`: keep as a legacy alias for
     `/gateway/api-keys`, but navigate to the LLM Connections section route (no General
     hash). Alternatively fold into the standalone route — retaining the alias is lower
     risk.

Simplest mechanical path: `git revert af7b02e62` (the "Move LLM connections into General"
commit) and re-apply only the model-allowlist wording tweak from `fe944a790` if that
commit mixed wording+placement. Verify the model-allowlist feature commits stay intact.

## Blast radius

- **Routes/redirects:** `/settings/llm-connections` becomes a real tab again;
  `/gateway/api-keys` stays redirected (to the standalone route now). No broken URLs —
  both were already handled, we're just changing the redirect *target*.
- **Deep links:** two in-app links (`EditEndpointFormRenderer`, `GenAIModelSelection`)
  switch from `general#llm-connections` back to the section route. Once the standalone
  route exists again, the hash form is unnecessary.
- **Tests to update:**
  - `settings/SettingsPage.test.tsx` — the `/settings/llm-connections` test currently
    asserts the embed renders via the General redirect; update it to assert the
    standalone section renders directly.
  - `experiment-tracking/.../GenAIModelSelection.test.tsx` (lines ~256, ~271) — asserts
    `stringContaining('#llm-connections')`; change to expect the section route without
    the hash.
  - `gateway/components/api-keys/ApiKeysList.test.tsx` — check the 1-line diff on this
    branch; keep allowlist-related assertions, revert only placement-coupled ones.
- **i18n:** the "LLM Connections" sidebar + page-title messages are being restored
  (they existed pre-branch), so no new strings; run `yarn i18n:check`.

## Risks

- **Low.** The main risk is accidentally reverting the model-allowlist work while undoing
  the placement move, since one commit (`fe944a790`) bundled wording + multi-model
  support. Do a file-scoped revert of the IA files rather than a blanket
  `git revert` of that commit, and diff the `model-selector/` + `useAllowlistedModelPairs`
  files against HEAD afterward to confirm they're untouched.
- Cosmetic follow-up (optional, not blocking): if discoverability is still a concern,
  add a one-line "Manage LLM connections →" pointer row inside General that links to the
  dedicated tab. That gives a General-surface breadcrumb without embedding the table.
