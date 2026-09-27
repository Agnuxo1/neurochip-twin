# Video slot

Drop the demo video here as `teaser.mp4` (H.264 + AAC, 16:9, ideally under 25 MB so the site stays small).
Optional English captions: `teaser.en.vtt` (WebVTT).

Then set `VIDEO_FILE: "video/teaser.mp4"` (and, if present, `VIDEO_CAPTIONS: "video/teaser.en.vtt"`) in the `CONFIG`
block at the top of `../assets/app.js`. While `VIDEO_FILE` is empty the page (`../index.html`, section `#video`) shows a
placeholder and requests nothing, so the console stays clean. If the video is hosted elsewhere instead, set `CONFIG.VIDEO_URL` and the "Video" buttons link to it.
