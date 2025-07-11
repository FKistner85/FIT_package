# Configuration Driven Development

All AI generated code should avoid hard coded parameters. Instead, define values in `src/FIT_python/config.yaml` and access them via the helpers in `FIT_python.config`. This ensures behaviour can be changed without editing the source code.
