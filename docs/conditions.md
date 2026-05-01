# Conditions

Conditions are an ordered list of conditional rules that filter or modify parameter combinations after the grid has been generated. They run in order.

The legacy key `constraints` is still accepted as a synonym for `conditions`.

## Anatomy of a condition

| Field     | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `name`    | string | no       | Human-readable label (used in error messages). |
| `when`    | object | **yes**  | Trigger: a mapping of `parameter_name: matcher`. All listed parameters must match (AND). |
| `exclude` | object | no*      | Drop matching combinations: `parameter_name: [values]`. |
| `force`   | object | no*      | Override a parameter: `parameter_name: value`. Duplicate combinations after forcing are deduped. |
| `set`     | object | no*      | Inject extra Hydra overrides: `hydra.path: value`. Keys are arbitrary Hydra paths (not parameters); these win over `hydra.static_overrides`. |

*At least one of `exclude`, `force`, or `set` is required per condition.

## `when` matchers

Each value in `when` can take three forms:

| Form | Example | Meaning |
|------|---------|---------|
| scalar | `optimizer: sgd` | exact equality |
| list   | `optimizer: [sgd, momentum_sgd]` | OR — matches any element |
| operator map | `learning_rate: {gt: 0.01}` | comparison |

Operator maps must contain exactly one operator key:

| Operator | Argument | Meaning |
|----------|----------|---------|
| `eq` | scalar | equal |
| `ne` | scalar | not equal |
| `gt` / `ge` | scalar | greater than / ≥ |
| `lt` / `le` | scalar | less than / ≤ |
| `in` | list | value is in the list |
| `not_in` | list | value is not in the list |

Numeric comparisons (`gt`/`ge`/`lt`/`le`) only match if both sides are numbers.

!!! tip "List values"
    If a discrete parameter's `values` are themselves lists, use `eq:` to disambiguate from the OR-list form (e.g. `layers: {eq: [3, 5, 7]}`).

## Examples

### Filter combinations with `exclude`

```yaml
conditions:
  - name: sgd_family_no_high_lr
    when:
      optimizer: [sgd, momentum_sgd]
      learning_rate: {gt: 0.01}
    exclude:
      learning_rate: [0.05, 0.1]
```

### Pin a parameter with `force`

```yaml
conditions:
  - name: adamw_fixed_wd
    when:
      optimizer: adamw
    force:
      weight_decay: 0.01
```

### Inject conditional Hydra overrides with `set`

```yaml
conditions:
  - name: adamw_warmup
    when:
      optimizer: adamw
    set:
      scheduler.type: cosine
      scheduler.warmup_steps: 1000
```

`set` keys are arbitrary Hydra paths — they're not validated against the `parameters` mapping, so this is the right place for non-parameter Hydra config.

## Validation

References in `when`, `exclude`, and `force` are validated at parse time against the `parameters` mapping. `set` keys are intentionally **not** validated — they're meant for non-parameter Hydra paths.

## Override ordering

Hydra applies overrides left-to-right; last wins. The override string is built in this order:

1. `experiment_name=<name>`
2. swept parameter `name=value` pairs
3. `hydra.static_overrides`
4. condition `set` extras

So `set` overrides `static_overrides`, which is the right shape for: *"these are the global defaults; conditionally adjust them based on the trial."*
