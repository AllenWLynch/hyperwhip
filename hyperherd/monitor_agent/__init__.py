"""Autonomous monitor daemon — the engine behind `herd monitor`.

A Python daemon built on the Claude Agent SDK that operates a HyperHerd
sweep on the user's behalf: staged rollout, failure triage, two-way
Discord channel for control. Opt-in via `pip install hyperherd[monitor]`.
"""
