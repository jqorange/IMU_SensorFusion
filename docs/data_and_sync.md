# Data Preparation and Video Synchronization

## Aligned IMU Input

The fusion pipeline accepts a nine-axis signal whose first sample already
corresponds to the target behavior or video start time. It does not search day
or session directories during fusion.

For legacy data, calculate the crop offset as:

```text
target_start_datetime - ephys_folder_datetime
```

Resample the continuous signal from 1250 Hz to 100 Hz before applying this
offset. This matches the ordering in the original batch workflow and prevents
rounding differences from selecting a different sample.

## Timestamp MAT Files

The preferred viewer timestamp variable is a one-dimensional
`video_timestamps_s` array. Its length should equal the number of readable
video frames, its unit is seconds, and its first value should be zero.

The viewer also accepts MATLAB datenums and datetime strings. Absolute values
are converted to elapsed time by subtracting the first timestamp. Automatic
selection favors arrays whose names contain `video` or `timestamp` and whose
length is close to the video frame count. Set `TIMESTAMP_KEY` when the MAT file
contains several plausible arrays.

Behavior sample times must not be treated as frame timestamps unless a verified
one-to-one mapping exists.

The viewer's `Offset (s)` control changes display alignment only. After finding
a correct offset, update the preparation crop so the stored data remains
reproducible.

## Large-File Strategy

Raw analog input is memory-mapped, and only the requested interval plus filter
padding is read. Prepared and fused arrays use chunked, compressed HDF5.

The viewer loads only the fusion time, Euler, and aligned quaternion datasets.
Video decoding runs in a background thread. Resized RGB frames are retained in
a bounded LRU memory cache, nearby frames are prefetched, and stale decode
requests are discarded during rapid seeking. Playback follows MAT timestamps
and skips obsolete frames instead of accumulating latency.
