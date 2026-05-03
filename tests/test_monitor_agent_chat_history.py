"""Tests for the chat-history rolling buffer.

The buffer captures real conversation (agent `msg` posts and user inbox
messages) but excludes per-tick heartbeats (`tick_summary`). It's stored
at .hyperherd/chat-history.jsonl and trimmed to a small fixed size so
the agent can stitch its past questions to the user's replies across
ticks without prompt bloat.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from hyperherd.monitor_agent import state as state_mod
from hyperherd.monitor_agent.tools import (
    CHAT_HISTORY_FILENAME, CHAT_HISTORY_KEEP, record_chat_entry,
)


class TestChatHistoryRecording(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _read_history(self):
        path = self.workspace / ".hyperherd" / CHAT_HISTORY_FILENAME
        if not path.is_file():
            return []
        return [
            json.loads(ln) for ln in path.read_text().splitlines()
            if ln.strip()
        ]

    def test_record_round_trip(self):
        record_chat_entry(
            self.workspace,
            role="agent", text="Herd dog: hi", via="discord", author="Herd dog",
        )
        record_chat_entry(
            self.workspace,
            role="user", text="how's it going", via="discord", author="alice",
        )

        entries = self._read_history()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["role"], "agent")
        self.assertEqual(entries[0]["author"], "Herd dog")
        self.assertEqual(entries[1]["role"], "user")
        self.assertEqual(entries[1]["text"], "how's it going")

    def test_trims_to_keep_limit(self):
        """Once we exceed CHAT_HISTORY_KEEP, the oldest drop off."""
        for i in range(CHAT_HISTORY_KEEP + 5):
            record_chat_entry(
                self.workspace,
                role="agent", text=f"msg-{i}",
                via="discord", author="Herd dog",
            )

        entries = self._read_history()
        self.assertEqual(len(entries), CHAT_HISTORY_KEEP)
        # The last KEEP messages are the most recent ones; the earliest
        # should have been evicted.
        last_idx = CHAT_HISTORY_KEEP + 4  # we recorded 0..(K+4)
        first_kept = last_idx - CHAT_HISTORY_KEEP + 1
        self.assertEqual(entries[0]["text"], f"msg-{first_kept}")
        self.assertEqual(entries[-1]["text"], f"msg-{last_idx}")

    def test_drain_inbox_mirrors_to_chat_history(self):
        """When state.compute drains the inbox, each user message also
        lands in chat-history so the agent has cross-tick context."""
        # Seed an inbox with two user messages.
        inbox_path = self.workspace / ".hyperherd" / state_mod.INBOX_FILE
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        inbox_path.write_text(
            json.dumps({
                "timestamp": "2026-05-03T00:00:00",
                "source": "discord", "author": "alice", "text": "pause",
            }) + "\n" +
            json.dumps({
                "timestamp": "2026-05-03T00:00:01",
                "source": "discord", "author": "alice", "text": "actually go",
            }) + "\n"
        )

        # Drain — the helper itself doesn't need a snapshot.
        msgs = state_mod._drain_inbox(self.workspace)
        self.assertEqual(len(msgs), 2)

        history = self._read_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["text"], "pause")
        # Timestamps are preserved from the inbox entries.
        self.assertEqual(history[0]["timestamp"], "2026-05-03T00:00:00")
        # The inbox file is truncated.
        self.assertEqual(inbox_path.read_text(), "")


class TestStateReadsChatHistory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_read_chat_history_parses_entries(self):
        record_chat_entry(self.workspace, role="agent",
                          text="Herd dog: starting", via="discord",
                          author="Herd dog")
        record_chat_entry(self.workspace, role="user",
                          text="thanks", via="discord", author="alice")

        entries = state_mod._read_chat_history(self.workspace)
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].role, "agent")
        self.assertEqual(entries[1].role, "user")
        self.assertEqual(entries[1].author, "alice")


if __name__ == "__main__":
    unittest.main()
