# Claude Code skills

HyperHerd ships **two** [Claude Code skills](https://docs.claude.com/en/docs/claude-code/skills):

- **`hyperherd-config`** — teaches Claude how to author and edit `hyperherd.yaml` files. Open a Claude Code session and ask "set up a sweep over learning rate and three optimizers" — Claude uses the skill's checklist + patterns to produce a config that follows HyperHerd conventions.
- **`hyperherd-monitor`** — the playbook for the [autonomous monitor](monitor.md): staged rollout, failure-triage policy, NaN/inf detection, status-report cadence. You don't invoke this skill by hand — `herd monitor` does that for you. Edit it if you want to change how the agent operates.

## Install

```bash
herd install-skill
```

That installs both skills into `~/.claude/skills/hyperherd-{config,monitor}/`. Open a new Claude Code session anywhere and they're available.

| Flag | Description |
|------|-------------|
| `--scope user` *(default)* | Writes to `~/.claude/skills/<skill>/`. Available in every Claude Code session. |
| `--scope project` | Writes to `./.claude/skills/<skill>/` in the current directory. Commit it and collaborators in this repo get the skill. |
| `--name <skill>` | Install only the named skill (`hyperherd-config` or `hyperherd-monitor`). Default installs both. |
| `-f, --force` | Overwrite an existing install. Use this after upgrading HyperHerd to pick up skill updates. |

### Re-installing after a HyperHerd upgrade

Both skills are shipped as package data inside the `hyperherd` Python package. After upgrading HyperHerd, run `herd install-skill --force` to pick up any updates.

## What `hyperherd-config` does

The config skill is a structured prompt that:

1. Walks Claude through the **sweep shape** decision (full grid / partial grid / one-at-a-time) and the implications for `default` fields.
2. Documents the parameter schema (discrete vs continuous, `abbrev`, `labels`, log scale).
3. Provides patterns for the four condition forms (`exclude`, `force`, `set`, `when` matchers).
4. Gives launcher templates for common environments.
5. Points Claude at the [Sweep config reference](configuration.md) for any field it's unsure about.

It is intentionally a **checklist + patterns**, not a substitute for the docs. For non-trivial configs Claude is instructed to read the full reference first.

## What `hyperherd-monitor` does

The skill defines the agent's behavior — staged rollout, failure-triage policy, NaN/inf kill rules, cadence, status-report format. See the [Autonomous monitor](monitor.md) page for what that adds up to from a user perspective.

If you want to change how the agent operates (different cadence, different triage policy, custom warnings), edit `~/.claude/skills/hyperherd-monitor/SKILL.md` directly. The skill is plain Markdown.

## Source

The canonical skill sources live in the HyperHerd repo at `hyperherd/skills/<name>/SKILL.md`. The repo's own `.claude/skills/<name>/SKILL.md` files are symlinks to them, so working in a checkout of the HyperHerd repo Just Works without running `install-skill`.
