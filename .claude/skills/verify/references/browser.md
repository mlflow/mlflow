# Browser checks and evidence

Use `--input-mode human` on local MLflow URLs, and initialize each session on its first `open`:

```bash
agent-browser --session verify --input-mode human \
  --init-script "$trusted_checkout/.claude/scripts/browser-init.js" open <url>
```

## Screenshots

Capture a screenshot when it shows the result of a check, and look at it before citing it:

```bash
agent-browser --session verify screenshot "$tmp_dir/out/media/<name>.png"
```

## Recordings

Record a flow when a still image cannot show it, such as a multi-step interaction.

1. Rehearse with stable selectors and dismiss overlays. Refs like `@e12` do not persist across
   sessions, so don't use them in the steps you save.
2. Save the commands to `$tmp_dir/work/steps.json`, e.g. `[["click", "<selector>"], ["wait", "500"]]`.
3. Close the rehearsal session, reset data if the flow changed it, and reopen a fresh session
   with the `open` command above.
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
