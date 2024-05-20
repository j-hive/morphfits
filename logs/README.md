# Logging
This directory contains logs for MorphFITS.

The program is currently configured to use a rotating file handler. This means
that once log files reach 1 million bytes, further records are recorded to the
next log file. Once four log files are full, the logger rotates back to the
first log file.

## Formatting
The logger formats lines as such:

```
yyyy-mm-ddThh:mm:ss LEVEL   MODULE       Message
```

where `MODULE` indicates a simplified display of the source of the log message,
e.g. `CONFIG`.