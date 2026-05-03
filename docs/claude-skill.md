# Claude Code skill

HyperHerd ships a [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills) called **`hyperherd-config`** that teaches Claude how to author and edit `hyperherd.yaml` files. Open a Claude Code session and ask "set up a sweep over learning rate and three optimizers" — Claude uses the skill's checklist + patterns to produce a config that follows HyperHerd conventions.

The autonomous monitor (`herd monitor`) ships its own playbook bundled inside the `hyperherd` Python package; it doesn't use the skill system.

## Install

```bash
herd install-skill
```

That installs the skill into `~/.claude/skills/hyperherd-config/`. Open a new Claude Code session anywhere and it's available.

| Flag | Description |
|------|-------------|
| `--scope user` *(default)* | Writes to `~/.claude/skills/hyperherd-config/`. Available in every Claude Code session. |
| `--scope project` | Writes to `./.claude/skills/hyperherd-config/` in the current directory. Commit it and collaborators in this repo get the skill. |
| `--name <skill>` | Install only the named skill. Currently only `hyperherd-config` ships, so this flag is mostly future-proofing. |
| `-f, --force` | Overwrite an existing install. Use this after upgrading HyperHerd to pick up skill updates. |

### Re-installing after a HyperHerd upgrade

The skill is shipped as package data inside the `hyperherd` Python package. After upgrading HyperHerd, run `herd install-skill --force` to pick up any updates.

## What the skill does

The skill is a structured prompt that:

1. Walks Claude through the **sweep shape** decision (full grid / partial grid / one-at-a-time) and the implications for `default` fields.
2. Documents the parameter schema (discrete vs continuous, `abbrev`, `labels`, log scale).
3. Provides patterns for the four condition forms (`exclude`, `force`, `set`, `when` matchers).
4. Gives launcher templates for common environments.
5. Points Claude at the [Sweep config reference](configuration.md) for any field it's unsure about.

It is intentionally a **checklist + patterns**, not a substitute for the docs. For non-trivial configs Claude is instructed to read the full reference first.

## Source

The canonical skill source lives in the HyperHerd repo at `hyperherd/skills/hyperherd-config/SKILL.md`. The repo's own `.claude/skills/hyperherd-config/SKILL.md` is a symlink to it, so working in a checkout of the HyperHerd repo Just Works without running `install-skill`.
