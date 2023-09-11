# FastAPI app for SSE streaming

import asyncio

from fastapi import FastAPI
from starlette.responses import HTMLResponse, StreamingResponse

app = FastAPI()


@app.get("/stream")
async def stream():
    async def event_stream():
        for data in ["foo", "bar", "baz"] * 3 + ["[DONE]"]:
            yield f"data: {data}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def read_items():
    return """
<html>
  <head>
    <title>Let's stream!</title>
  </head>
  <body>
    <script>
      var eventSource = new EventSource("/stream");
      eventSource.onmessage = function (e) {
        var stream = document.getElementById("stream");
        stream.innerHTML = stream.innerHTML + e.data + "<br>";
        if (e.data == "[DONE]") {
          eventSource.close();
        }
      };
    </script>
    <h1>Let's stream!</h1>
    <ul id="stream"></ul>
  </body>
</html>
"""
