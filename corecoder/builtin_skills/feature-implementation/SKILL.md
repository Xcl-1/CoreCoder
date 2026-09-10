# Goal

Implement the requested behavior completely with the smallest coherent change.

# Workflow

1. Locate the relevant entry points, callers, tests, and public contracts.
2. Translate the request into observable acceptance criteria.
3. Implement the feature using existing project conventions and abstractions.
4. Add or update focused tests for the new behavior and important edge cases.
5. Run the narrowest meaningful verification, then check broader regressions when justified.

# Bounded-task fast path

- When the request names the implementation file and tests, read those named files in one parallel tool round. Do not glob or search for paths already supplied.
- If an explicit stub and complete tests define the behavior, implement directly after that read; do not spend a separate tool round proving that the stub fails.
- Batch independent reads together, make one coherent edit, and run the narrowest relevant verification once as one direct command. The runtime already starts commands in the workspace; never prepend `cd`, chain commands, or redirect output.
- When that verification passes, reason about any remaining requested edge cases from the source and tests. Do not launch an extra ad-hoc Python command, perform another repository survey, or repeat the same test; return the final report immediately.

# Boundaries

- Preserve unrelated behavior and avoid speculative extensions.
- Do not silently change public interfaces, configuration, or persisted formats.
- Report any requested part that could not be verified.
