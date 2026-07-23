# LLM Connections Redesign — Implementation Plan

> Companion to `docs/design/llm-connections-redesign.md`. Grounds the approved design in the real MLflow OSS codebase (`/Users/joshua.wong/mlflow-scratch`). All file:line references verified against the working tree.
>
> **Scope:** Phase 1 only — IA move into General, first-class per-key model allowlist (persistence + API + add/edit UI), and the allowlisted-pair dropdown on **Detect Issues** (`IssueDetectionModal` / `GenAIModelSelection`). Assistant and scorers surfaces are **explicitly deferred to Phase 2**.

---

## 0. Current-state map (what exists today)

**Settings IA**
- `settings/settingsSectionConstants.ts:1-25` — three path segments: `general`, `llm-connections`, `webhooks`. `SETTINGS_PATH_SEGMENTS` + `isSettingsPathSegment` type-guard.
- `settings/SettingsPage.tsx:85-101` — `activeSection` derived from `:section` param; unknown sections `navigate(...replace)` to General (preserving `returnTo`). The `llm-connections` branch at `SettingsPage.tsx:272-290` renders a `SettingsSectionHeader` + `<ApiKeysPageInner />`.
- `common/components/MlflowSidebarSettingsItems.tsx:71-79` — the sub-sidebar link for the LLM Connections tab.
- `experiment-tracking/route-defs.ts:278-281` — page-title case for `llm-connections`.
- `gateway/pages/RedirectApiKeysToSettings.tsx:13` — legacy `/gateway/api-keys` redirects to the `llm-connections` section route.
- `gateway/components/edit-endpoint/EditEndpointFormRenderer.tsx:524` — deep-links into `llm-connections`.
- Route base: `experiment-tracking/routes.ts:190` `getSettingsSectionRoute(section)`.

**The manager being moved**
- `gateway/pages/ApiKeysPage.tsx` — `ApiKeysPageInner` composes `ApiKeysList` + `CreateApiKeyModal` + `ApiKeyDetailsDrawer` + endpoint/binding drawers, wired via `useApiKeysPage`.
- Components dir: `gateway/components/api-keys/` (`ApiKeysList.tsx`, `CreateApiKeyModal.tsx`, `ApiKeyDetailsDrawer.tsx`, `EditApiKeyModal.tsx`, `DeleteApiKeyModal.tsx`, ...).

**Secrets domain (types / API)**
- `gateway/types.ts:5-18` `ProviderModel` (`model`, `provider`, capability flags). `:55-96` `SecretInfo`, `CreateSecretRequest`, `UpdateSecretRequest` all carry `auth_config?: Record<string, string>`.
- `gateway/api.ts:112-159` — `createSecret` / `getSecret` / `updateSecret` / `deleteSecret` / `listSecrets` against `ajax-api/3.0/mlflow/gateway/secrets/*`.
- Reusable pickers: `gateway/components/model-selector/ModelSelectorModal.tsx` (`onSelect(model: ProviderModel)`, `provider`, `initialValue`), `gateway/components/create-endpoint/ModelSelect.tsx` (single-value wrapper around the modal), `.../create-endpoint/ProviderSelect.tsx`.

**Consuming surface (Detect Issues)**
- `.../traces-v3/IssueDetectionModal.tsx` — 2-step modal; step 2 hosts `<GenAIModelSelection>` via a `ref`; `handleSubmit` (`:93-152`) reads `getValues()` → `{ mode, endpointName, provider, model, apiKeyConfig, saveKey }`, optionally `createSecret`, then `invokeIssueDetection({ provider, model, secret_id, endpoint_name, ... })`.
- `.../traces-v3/GenAIModelSelection.tsx` — the inline provider+model+key entry (endpoint dropdown, `ProviderSelect`-style combobox, `ModelSelect`, `GenAIApiKeyConfigurator`, advanced settings). Exposes `ModelSelectionValues` + imperative `getValues/reset/isValid`.

**Backend**
- Entity: `mlflow/entities/gateway_secrets.py` — `GatewaySecretInfo` (frozen dataclass), `auth_config: dict[str,Any] | None` at `:50`; `to_proto`/`from_proto` at `:58-89`.
- Proto: `mlflow/protos/service.proto:5212-5416` — `GatewaySecretInfo` has `map<string,string> auth_config = 9` (`:5232`); `CreateGatewaySecret.auth_config = 5` (`:5358`), `UpdateGatewaySecret.auth_config = 4` (`:5391`). **All are `map<string,string>` — values are strings only.** Field 4 in create is `reserved`; there is room for a new field.
- Store: `mlflow/store/tracking/gateway/sqlalchemy_mixin.py:169-236` (`create_gateway_secret`), `:266-325` (`update_gateway_secret`), `:238-264` (`get_secret_info`), `:347+` (`list_secret_infos`). `auth_config` serialized via `json.dumps(...)` into a `Text` column.
- Model: `mlflow/store/tracking/dbmodels/models.py:2398-2516` `SqlGatewaySecret`; `auth_config = Column(Text, nullable=True)` at `:2453`; workspace column + `uq_secrets_workspace_secret_name` at `:2480-2494`; `to_mlflow_entity` at `:2499`.
- Handlers: `mlflow/server/handlers.py` — `_create_gateway_secret` (~5660), `_get_gateway_secret_info` (~5688), `_update_gateway_secret` (~5704), `_list_gateway_secrets` (~5745). Note existing pattern `dict(request_message.auth_config) or None`.
- Consumption path: `_invoke_issue_detection_handler` (`handlers.py:4925-4970`) → `_fetch_provider_credentials(store, provider, secret_id)` (`mlflow/genai/discovery/job.py:12-43`). **Critical:** `job.py:32` does `secret_dict = secret_value | auth_config` then maps into env vars — so anything stuffed into `auth_config` bleeds into credential resolution.
- Tests: gateway store tests with workspace-enabled/disabled parametrization already live at `tests/store/tracking/test_gateway_sql_store.py:112-126` (fixture `workspaces_enabled`, `WorkspaceContext`, `DEFAULT_WORKSPACE_NAME`). The path `tests/store/tracking/test_sqlalchemy_store_workspace.py` **does not exist** — the correct home for workspace-aware secret tests is `test_gateway_sql_store.py` (see CLAUDE.md note in the Risks section).

---

## 1. Key design question: where do allowlisted models live?

**The candidate: piggyback on the existing `auth_config` JSON blob.**

### Why it's tempting
- Zero schema migration — `auth_config` is already a `Text` JSON column (`models.py:2453`), already round-tripped through create/get/update/list, already surfaced on `SecretInfo` (both TS and proto).

### Why it's the wrong home (recommendation: **do NOT reuse `auth_config`**)
1. **Type mismatch at the proto boundary.** `auth_config` is `map<string,string>` (`service.proto:5232/5358/5391`). Allowlisted models are a *list of structured objects* (`provider` + `model`, ideally capability flags). You'd have to JSON-`stringify` a list into one string value inside the map — a nested-encoding hack that every reader must know to decode, and which defeats proto/dataclass typing.
2. **Semantic collision / credential leakage.** `auth_config` is defined as *provider auth metadata* (region, project_id, `auth_mode`) and is **merged into the credential dict** at `job.py:32` (`secret_value | auth_config`) before env-var mapping. Injecting an `allowlisted_models` key risks it being treated as a credential field and is a genuine correctness/security smell. `auth_config` should stay "things the provider SDK needs to authenticate."
3. **Query/immutability constraints.** The allowlist is a first-class concept the UI lists, filters, and (Phase 2) other surfaces read. Burying it in an opaque per-provider blob makes it un-queryable and couples it to auth semantics.

### Recommendation: **dedicated persistence, expressed as a first-class field on the secret.**

Two viable shapes; pick **A** for Phase 1:

**Option A (recommended) — new `allowlisted_models` column on `secrets` (JSON Text), first-class proto field.**
- Add `allowlisted_models = Column(Text, nullable=True)` to `SqlGatewaySecret` (JSON-encoded list of `{provider, model}`), mirroring exactly how `auth_config` is already handled (`json.dumps`/`json.loads`) — but as a *separate, purpose-named* column that never touches credential resolution.
- Add a repeated message field to the proto (new field numbers on `GatewaySecretInfo`, `CreateGatewaySecret`, `UpdateGatewaySecret`) — a small `GatewayAllowlistedModel { string provider; string model; }` message, `repeated` on each. This keeps typing honest and leaves `auth_config` untouched.
- Requires one Alembic migration (add nullable column) — cheap, additive, backward-compatible (existing rows → `NULL` → empty allowlist).

**Option B — separate `secret_allowlisted_models` join table.**
- Cleaner normalization, enables per-model metadata/uniqueness constraints and future per-model settings.
- More code (new SqlAlchemy model, migration, CRUD), heavier than Phase 1 needs. **Defer unless** product wants per-model attributes beyond `{provider, model}`.

**Tradeoff summary**

| | A: new JSON column | B: join table | (rejected) reuse `auth_config` |
|---|---|---|---|
| Migration | 1 additive column | new table | none |
| Typing | honest (repeated proto msg) | honest | nested string hack |
| Credential-path safety | isolated | isolated | ❌ leaks into `job.py:32` |
| Query/normalize | limited | full | none |
| Effort | low | medium | trivial-but-wrong |

**Go with Option A.** It matches the existing `auth_config` serialization pattern the team already maintains, keeps the concept first-class and typed, and is a single additive migration.

---

## 2. Ordered implementation steps

Each step lists a verifiable outcome and the exact command(s) to run. Frontend commands run from repo root via `pushd mlflow/server/js && yarn <cmd>; popd`.

### Step 1 — Backend persistence (store + model + entity + migration)
**Changes**
- `models.py:2453` area — add `allowlisted_models = Column(Text, nullable=True)` to `SqlGatewaySecret`; update `to_mlflow_entity` (`:2499`) to `json.loads` it into the entity.
- `mlflow/entities/gateway_secrets.py` — add `allowlisted_models: list[dict] | None = None` (or a small typed `GatewayAllowlistedModel` dataclass list), update `to_proto`/`from_proto` (`:58-89`).
- `sqlalchemy_mixin.py` — thread the field through `create_gateway_secret` (`:169`), `update_gateway_secret` (`:266`, following the same "None = unchanged, `[]` = clear" convention already used for `auth_config` at `:314-316`), and ensure `get_secret_info`/`list_secret_infos` return it.
- New Alembic migration under `mlflow/store/db_migrations/versions/` (model after `1bd49d398cd23_add_secrets_tables.py`): add the nullable `allowlisted_models` column. Update the migration head chain.
- Leave `_fetch_provider_credentials` (`job.py`) **untouched** — the allowlist must not enter credential resolution.

**Verify**
```bash
uv run pytest tests/store/tracking/test_gateway_sql_store.py -q
# migration applies cleanly (fresh sqlite):
uv run pytest tests/store/db/ -q -k migration   # or the repo's migration test module
```
Outcome: create/get/update/list round-trip `allowlisted_models`; existing rows return `None`/empty.

### Step 2 — Proto + API surface (backend handlers)
**Changes**
- `service.proto` — add `message GatewayAllowlistedModel { optional string provider = 1; optional string model = 2; }`; add `repeated GatewayAllowlistedModel allowlisted_models = 10;` to `GatewaySecretInfo`, and new repeated fields to `CreateGatewaySecret` and `UpdateGatewaySecret` (reuse reserved slots where available). Regenerate `service_pb2` per repo proto-gen instructions.
- `handlers.py` — `_create_gateway_secret` / `_update_gateway_secret` extract and pass `allowlisted_models` (mirroring the `auth_config` extraction pattern). Add `_assert_array`-style validation.

**Verify**
```bash
uv run pytest tests/server/ -q -k "gateway_secret or secret"
uv run pytest tests/store/tracking/test_gateway_sql_store.py -q
```
Outcome: REST create/update accepts `allowlisted_models`; get/list echo it back.

### Step 3 — Settings IA move (redirect old tab into General)
**Changes**
- `SettingsPage.tsx` — move the `ApiKeysPageInner` render out of the `llm-connections` branch and into the `general` branch as a titled section (H2 `SettingsSectionHeader` + helper text, `id="llm-connections"` anchor for deep-link scroll). Delete/short-circuit the `SETTINGS_SECTION_LLM_CONNECTIONS` branch (`:272-290`).
- Add anchor-scroll: on mount, if `location.hash === '#llm-connections'`, scroll the section into view (use a `ref` + `scrollIntoView`, respecting `reduced-motion`).
- Redirect the standalone route: in the `useEffect` at `SettingsPage.tsx:92-101`, treat `llm-connections` like an unknown/legacy section and `navigate(...General + '#llm-connections', {replace:true})` preserving `returnTo`.
- `settingsSectionConstants.ts` — **keep** `SETTINGS_SECTION_LLM_CONNECTIONS` exported as a legacy constant (for the redirect + existing deep-links in `EditEndpointFormRenderer.tsx:524` and `RedirectApiKeysToSettings.tsx:13`), but **remove it from `SETTINGS_PATH_SEGMENTS`** so it's no longer a first-class tab. Alternatively point those deep-links at `getSettingsSectionRoute(GENERAL) + '#llm-connections'` via a new helper.
- `MlflowSidebarSettingsItems.tsx:71-79` — remove the LLM Connections sub-sidebar link (now a section within General).
- `route-defs.ts:278-281` — collapse the `llm-connections` page-title case (redirect makes it General).
- `RedirectApiKeysToSettings.tsx` and `EditEndpointFormRenderer.tsx:524` — retarget to `General#llm-connections`.

**Verify**
```bash
pushd mlflow/server/js && yarn test SettingsPage; popd
pushd mlflow/server/js && yarn type-check; popd
```
Manual: visiting `/settings/llm-connections` and `/gateway/api-keys` lands on General scrolled to the connections section; the Detect-Issues "manage keys" tooltip/link (`GenAIModelSelection.tsx:537-539`) still resolves.

### Step 4 — Add/Edit-connection model-allowlist UI
**Changes**
- `gateway/types.ts` — add `allowlisted_models?: ProviderModel[]` (or `{provider,model}[]`) to `SecretInfo`, `CreateSecretRequest`, `UpdateSecretRequest` (`:55-96`).
- `gateway/api.ts` — no signature change needed (payloads are pass-through objects); confirm the field serializes in `createSecret`/`updateSecret` bodies (`:112-138`).
- `CreateApiKeyModal.tsx` / `EditApiKeyModal.tsx` (in `gateway/components/api-keys/`) — add a **Step 2 "Allowed models"** collector. **New UI = a thin multi-select chip collector** that wraps the existing `ModelSelectorModal` (filtered to the chosen provider) and accumulates `ProviderModel[]` as removable chips (design §4). Enforce "≥1 model to save" with an inline message (design §4, not a silent disabled button).
- `ApiKeysList.tsx` row — render allowlisted models as chips (design §3).

**Reusable vs new**
- Reuse as-is: `ProviderSelect`, `ModelSelectorModal` (`onSelect(model)`), `ModelSelect`, `useModelsQuery`, secret form fields, `useApiKeyConfiguration`.
- Genuinely new: the multi-model chip collector (accumulate + remove), the "≥1 model" validation, chip rendering on the list row.

**Verify**
```bash
pushd mlflow/server/js && yarn test CreateApiKeyModal ApiKeysList; popd
pushd mlflow/server/js && yarn type-check && yarn i18n:check; popd
```
Outcome: a saved connection persists its allowlist; list row shows chips; edit round-trips.

### Step 5 — Consuming-surface allowlisted-pair dropdown (Detect Issues only)
**Changes**
- Build the flat option list = every `provider · model` pair across all connections. Source it from `listSecrets()` + each secret's `allowlisted_models` (new `useAllowlistedModelPairs` hook, colocated with gateway hooks).
- In `GenAIModelSelection.tsx`, gate the new single-dropdown mode behind the existing `showConfigureDirectly` path (or a new prop) so Phase 2 surfaces are untouched. The dropdown selects a pair and derives `{ provider, model, secret_id }`; keep the `getValues()` contract shape (`ModelSelectionValues`) so `IssueDetectionModal.handleSubmit` (`:93-152`) needs **no** backend-contract change — it still sends `provider`/`model`/`secret_id`.
- Dropdown must include: active-pair highlight (`nav-state-active`), empty state with "＋ Add a connection →" deep-linking to `General#llm-connections`, and a "Manage connections →" footer (design §5).
- Backend `_invoke_issue_detection_handler` (`handlers.py:4925`) and `job.py` are **unchanged** — the surface still supplies `provider`+`model`+`secret_id`.

**Verify**
```bash
pushd mlflow/server/js && yarn test GenAIModelSelection IssueDetectionModal; popd
pushd mlflow/server/js && yarn lint && yarn prettier:check && yarn i18n:check && yarn type-check; popd
```
Manual (dev server per CLAUDE.md): Detect Issues → single dropdown of `provider · model` pairs → Run Analysis succeeds; empty state deep-links to the connections section.

### Step 6 — Full pre-commit gate
```bash
pushd mlflow/server/js && yarn lint && yarn prettier:check && yarn i18n:check && yarn type-check; popd
uv run pytest tests/store/tracking/test_gateway_sql_store.py tests/server -q -k "secret or gateway or issue_detection"
```

---

## 3. i18n, type-checks, workspace tests

- **i18n:** every new string uses `FormattedMessage` / `intl.formatMessage` with `defaultMessage` + `description` (see existing `SettingsPage.tsx`, `GenAIModelSelection.tsx`). Run `yarn i18n:check`. New strings: section header/helper, "Allowed models", "Add model", "≥1 model required" validation, dropdown empty state, "Manage connections".
- **Type checks:** `yarn type-check` after every FE step; keep `SecretInfo`/request types in sync with proto.
- **Workspace-aware tests (CLAUDE.md requirement):** the design prompt names `tests/store/tracking/test_sqlalchemy_store_workspace.py`, but that file does not exist. The correct, existing home is **`tests/store/tracking/test_gateway_sql_store.py`**, which already parametrizes `workspaces_enabled` (`:112-126`) via `WorkspaceContext`. Add allowlist create/get/update/list assertions there so they run under both workspace-enabled and workspace-disabled params. Confirm `SqlGatewaySecret`'s workspace column + `uq_secrets_workspace_secret_name` (`models.py:2480-2494`) are unaffected by the additive column.

---

## 4. Risks & backward-compat

- **Existing secrets with no allowlist.** New column is `NULL` → entity `allowlisted_models=None` → dropdown treats them as "no allowlisted models." These keys become non-selectable in the new one-pick dropdown. **Mitigation:** the connections list must surface an "Add models" affordance on legacy rows (design §3 `[+ model]`), and Detect Issues must not silently hide keys — show the empty/"add a connection" state. Do **not** auto-backfill.
- **Proto field numbering.** Reuse reserved slots carefully; never renumber existing fields. `auth_config` field numbers (`5/4/9`) stay put.
- **Do not route the allowlist through `_fetch_provider_credentials`** (`job.py:32`). Keep it a display/selection concept only in Phase 1.
- **Deep-link breakage.** `/settings/llm-connections`, `/gateway/api-keys`, and `EditEndpointFormRenderer.tsx:524` must all resolve post-move. Covered by Step 3 redirect + retarget; add a redirect test.
- **Migration on existing DBs** (sqlite + the server-supported backends). Column must be nullable with no server default beyond `NULL`; test upgrade + downgrade.
- **Immutability constraint** on `secret_id`/`secret_name` (AAD, `gateway_secrets.py:18-22`) — the allowlist column is unrelated to encryption AAD, so editing it is safe (unlike renaming). Confirm no test asserts frozenness of the whole entity in a way the new field violates.
- **`ApiKeysList.test.tsx` / `CreateApiKeyModal` existing snapshots** may need updating for the new chips/step.

## 5. Deferred (Phase 2 — do NOT build now)
- Assistant and scorers consuming surfaces adopting the allowlisted-pair dropdown.
- Join-table normalization (Option B) / per-model metadata.
- Key-validation-on-blur against the gateway and status pills (design §3/§4) beyond what already exists — nice-to-have, not required for the IA move + allowlist MVP.
