---
name: upload-media
description: Upload one or more local images or videos to GitHub and get back a `user-attachments` URL for each, to embed in a PR body, issue, or comment. Use when asked to attach screenshots or screen recordings.
argument-hint: "path(s) to the image or video to upload"
---

# Upload media

Files: $ARGUMENTS (when empty, the paths named in the request).

```bash
uv run --package skills skills upload-media <path>...  # prints "<path>\t<url>" per file
```

Embed an image as `![alt](url)`. Embed a video as the bare URL in its own paragraph, blank line above and below; anything else renders as a link rather than a player.

No GitHub documentation covers this endpoint, so it can stop working without notice. Source: <https://x.com/steipete/status/2088486859244741020>.
