from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "base"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dysonspherain.token_economy.tokenizer_calibration import calibrate
from dysonspherain.utils.token_counter import TokenCounter


class TokenizerCalibrationTests(unittest.TestCase):
    def test_mixed_content_heuristic_classifies_common_inputs(self) -> None:
        counter = TokenCounter(strategy="mixed_content_heuristic")
        samples = {
            "zh_text": "这是一个用于测试中文 token 估算的段落。",
            "code": "def run(x):\n    return x + 1\n",
            "json": '{"status": "ok", "items": [1, 2, 3]}',
            "markdown": "# Title\n\n- item\n- item",
            "log": "Traceback (most recent call last):\nValueError: bad",
        }
        for expected, text in samples.items():
            result = counter.count(text)
            self.assertGreater(result.tokens, 0)
            self.assertTrue(result.fallback_used)
            self.assertIn(expected if expected != "code" else "code", result.tokenizer_name)

    def test_calibration_file_can_be_written_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "samples.jsonl"
            output_path = root / "calibration.json"
            input_path.write_text(
                "\n".join(
                    [
                        json.dumps({"kind": "zh_text", "text": "中文样本", "reference_tokens": 8}, ensure_ascii=False),
                        json.dumps({"kind": "code", "text": "def f(): return 1"}),
                    ]
                ),
                encoding="utf-8",
            )
            payload = calibrate(input_path, output_path)
            self.assertEqual(payload["sample_count"], 2)
            self.assertTrue(output_path.exists())
            result = TokenCounter(strategy="calibrated", calibration_file=output_path).count("中文样本")
            self.assertGreater(result.tokens, 0)
            self.assertTrue(result.fallback_used)


if __name__ == "__main__":
    unittest.main()
