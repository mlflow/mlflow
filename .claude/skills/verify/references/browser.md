# Browser checks and evidence

Use `--input-mode human` on local MLflow URLs, and initialize each session on its first `open`:

```bash
agent-browser --session verify --input-mode human \
  --init-script "$trusted_checkout/.claude/scripts/browser-init.js" open <url>
```

## Themes

MLflow reads the theme from local storage when the page loads. The first load of a session
stores the system preference, so the starting theme varies and `set media dark` has no effect
afterwards. Set the key and reload for each theme, `true` for dark and `false` for light:

```bash
agent-browser --session verify storage local set _mlflow_dark_mode_toggle_enabled true
agent-browser --session verify reload
```

Capture each theme under its own name, such as `<name>-light.png` and `<name>-dark.png`, and
pass `"dark"` as the caption theme on dark pages.

## Evidence for findings

Capture every finding you reproduce in the browser and cite it in its inline comment. Before
choosing a screenshot, ask: would a reviewer who sees only this image understand why the state
is wrong? If the answer depends on something done earlier (a selection, a search, a page
change, a Back press), record a short video instead. For a regression, capture the same steps
on the base and the PR.

## Screenshots

Capture a screenshot when it shows the result of a check, and look at it before citing it:

```bash
agent-browser --session verify screenshot "$tmp_dir/out/media/<name>.png"
```

## Recordings

1. Rehearse with stable selectors, and hide side panels, popups, and onboarding hints that cover
   the page. Refs like `@e12` do not persist across sessions, so don't use them in the steps
   you save.
2. Save the commands to `$tmp_dir/work/steps.json`, e.g. `[["click", "<selector>"], ["wait", "500"]]`.
3. Close the rehearsal session, reset data if the flow changed it, and reopen a fresh session
   with the `open` command above. A fresh session loses the theme, so set it again before
   `record start`.
4. Replay while recording:

```bash
agent-browser --session verify record start "$tmp_dir/out/media/<name>.mp4" --cursor --contact-sheet
timeout 120 agent-browser --session verify batch --bail < "$tmp_dir/work/steps.json"
agent-browser --session verify record stop
```

Add captions and short waits around subtle actions such as Back or key presses, so a viewer can
follow them: `["eval", "caption('Click Save')"]` shows one, and `caption()` hides it.
`caption(text, theme, position)` takes an optional theme and position.

Inspect the contact sheet before citing a recording. If a replay times out, retry once in a
fresh session, then fall back to screenshots.
