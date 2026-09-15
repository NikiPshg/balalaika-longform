# Source licenses and attribution

The audio recordings in Balalaika-longform originate from YouTube videos marked
**Creative Commons Attribution (CC BY)** by their uploaders in the collected
source metadata. The recordings retain those source licenses; this dataset does
not replace their terms with a new blanket license.

For every sample, its JSON `source` object contains the video title, original URL,
channel attribution where available, and the archived licensing evidence.
`metadata/sources.jsonl` collects these records once per source video.

The saved YouTube `creativeCommon` flag and the human-readable Creative Commons
label do not specify a license version. The release records `license_version:
null` rather than inferring a version from current YouTube documentation.
Refer to the original source for its applicable declaration and attribution.

When redistributing a recording or an adaptation, preserve the source attribution
and license notices and indicate the modifications. Processing in this release
includes extracting speech segments, the recorded loudness-normalization step,
and DeepFilterNet3 speech enhancement. Transcripts are automatic annotations of
the source speech.

The archived uploader declarations have been checked for all included videos.
They are not an independent rights audit of underlying literary works or
performances and do not establish individual speaker consent for every possible
downstream use.

References:

- [YouTube: license types and Creative Commons attribution](https://support.google.com/youtube/answer/2797468?hl=en)
- [Creative Commons: Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)
- [Creative Commons: Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)

Please report attribution corrections or removal requests through the dataset's
[discussion page](https://huggingface.co/datasets/lab260/Balalaika-longform/discussions),
identifying the source video or sample.
