# Claude Code skill

HyperHerd ships a [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) that teaches Claude how to author and edit `hyperherd.yaml` files. With the skill installed, you can ask Claude things like:

- *"Set up a HyperHerd sweep over learning rate (log scale, 5 values from 1e-5 to 1e-2) and three optimizers."*
- *"Add a condition that excludes high learning rates when I'm using SGD."*
- *"Convert this sweep from full grid to one-at-a-time."*

…and Claude will use the skill's checklist + patterns to produce a config that follows HyperHerd conventions.

## Install

```bash
herd install-skill
```

This drops `SKILL.md` into `~/.claude/skills/hyperherd-config/`. Open a new Claude Code session and the skill is available everywhere.

### Project scope

If you'd rather scope the skill to a single repository (e.g. for collaborators to pick it up via the repo checkout), install it per-project:

```bash
herd install-skill --scope project
```

This writes to `./.claude/skills/hyperherd-config/SKILL.md` in the current directory. Commit that file and anyone working in the repo with Claude Code will get the skill.

### Re-installing after a HyperHerd upgrade

The skill is shipped as package data inside the `hyperherd` Python package. After upgrading HyperHerd, re-run `herd install-skill --force` to pick up any updates to the skill.

## What the skill does

The skill is a structured prompt that:

1. Walks Claude through the **sweep shape** decision (full grid / partial grid / one-at-a-time) and the implications for `default` fields.
2. Documents the parameter schema (discrete vs continuous, `abbrev`, `labels`, log scale).
3. Provides patterns for the four condition forms (`exclude`, `force`, `set`, `when` matchers).
4. Gives launcher templates for common environments.
5. Points Claude at the [Sweep config reference](configuration.md) for any field it's unsure about.

The skill is intentionally a **checklist + patterns**, not a substitute for the docs. For non-trivial configs Claude is instructed to read the full reference first.

## Source

The canonical skill source lives in the HyperHerd repo at `hyperherd/skill/SKILL.md`. The repo's own `.claude/skills/hyperherd-config/SKILL.md` is a symlink to it, so working in a checkout of the HyperHerd repo Just Works without running `install-skill`.
