"""Offline coverage for configuration identity and legacy checkpoint lookup."""
import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from paper2slides.core.paths import (
    get_base_dir,
    get_config_dir,
    get_config_name,
    get_mode_dir,
    get_plan_checkpoint,
    get_rag_checkpoint,
    get_summary_checkpoint,
)
from paper2slides.core.pipeline import list_outputs, run_pipeline
from paper2slides.core.state import (
    create_state,
    detect_start_stage,
    load_state,
    save_state,
)


class ConfigPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = get_base_dir(self.temp.name, "sample", "paper")
        self.config = {
            "output_type": "slides",
            "style": "custom",
            "custom_style": "Watercolor theme with warm reds",
            "slides_length": "medium",
            "fast_mode": True,
        }
        self.other = {**self.config, "custom_style": "Watercolor theme with cool blues"}
        self.legacy = get_mode_dir(self.base, self.config) / "slides_custom_Watercolor_theme_medium"

    def save_checkpoints(self, config_dir, config):
        for path in (
            get_rag_checkpoint(self.base, config),
            get_summary_checkpoint(self.base, config),
            get_plan_checkpoint(config_dir),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        state = create_state(config)
        state["stages"] = {stage: "completed" for stage in state["stages"]}
        save_state(config_dir, state)

    def test_custom_names_include_entire_unnormalized_style(self):
        for first, second in (
            (self.config["custom_style"], self.other["custom_style"]),
            ("blue/red", "blue red"),
            ("blue red", "blue_red"),
        ):
            with self.subTest(first=first, second=second):
                a = {**self.config, "custom_style": first}
                b = {**self.config, "custom_style": second}
                self.assertNotEqual(get_config_name(a), get_config_name(b))
                self.assertEqual(get_config_name(a), get_config_name(dict(a)))

    def test_predefined_and_empty_custom_names_are_unchanged(self):
        self.assertEqual(get_config_name({}), "slides_academic_medium")
        self.assertEqual(
            get_config_name({"style": "doraemon", "output_type": "poster", "poster_density": "dense"}),
            "poster_doraemon_dense",
        )
        for custom in (None, ""):
            with self.subTest(custom=custom):
                config = {**self.config, "custom_style": custom}
                self.assertEqual(get_config_name(config), "slides_custom_custom_medium")
                self.assertEqual(
                    get_config_dir(self.base, config),
                    get_mode_dir(self.base, config) / "slides_custom_custom_medium",
                )
        self.assertEqual(get_config_name({"style": "custom"}), "slides_custom_custom_medium")

    def test_exact_legacy_configuration_remains_resumable(self):
        self.save_checkpoints(self.legacy, self.config)
        self.assertEqual(get_config_dir(self.base, self.config), self.legacy)
        self.assertEqual(detect_start_stage(self.base, self.legacy, self.config), "generate")

    def test_legacy_configuration_must_match_full_identity(self):
        for changed in (
            {"custom_style": self.other["custom_style"]},
            {"style": "academic"},
            {"output_type": "poster"},
            {"slides_length": "long"},
            {"fast_mode": False},
        ):
            with self.subTest(changed=changed):
                self.save_checkpoints(self.legacy, {**self.config, **changed})
                state_path = self.legacy / "state.json"
                before = state_path.read_bytes()
                chosen = get_config_dir(self.base, self.config)
                self.assertNotEqual(chosen, self.legacy)
                self.assertEqual(chosen.name, get_config_name(self.config))
                self.assertFalse(chosen.exists())
                self.assertEqual(state_path.read_bytes(), before)

    def test_poster_legacy_lookup_matches_density(self):
        config = {**self.config, "output_type": "poster", "poster_density": "sparse"}
        legacy = get_mode_dir(self.base, config) / "poster_custom_Watercolor_theme_sparse"
        self.save_checkpoints(legacy, config)
        self.assertEqual(get_config_dir(self.base, config), legacy)
        save_state(legacy, create_state({**config, "poster_density": "dense"}))
        self.assertNotEqual(get_config_dir(self.base, config), legacy)

    def test_unverifiable_legacy_state_is_not_reused_or_modified(self):
        self.legacy.mkdir(parents=True)
        state_path = self.legacy / "state.json"
        self.assertNotEqual(get_config_dir(self.base, self.config), self.legacy)
        for contents in (
            b"{", b"\xff", b"[]", b"null", b"{}", b'{"config": []}',
            b'{"config": null}', b'{"config": {}}',
        ):
            with self.subTest(contents=contents):
                state_path.write_bytes(contents)
                chosen = get_config_dir(self.base, self.config)
                self.assertNotEqual(chosen, self.legacy)
                self.assertFalse(chosen.exists())
                self.assertEqual(state_path.read_bytes(), contents)

    def test_expected_read_error_does_not_reuse_legacy_state(self):
        self.save_checkpoints(self.legacy, self.config)
        with patch.object(Path, "read_text", side_effect=PermissionError("synthetic unreadable state")):
            self.assertNotEqual(get_config_dir(self.base, self.config), self.legacy)

    def test_existing_new_directory_has_priority_over_legacy(self):
        self.save_checkpoints(self.legacy, self.config)
        current = get_mode_dir(self.base, self.config) / get_config_name(self.config)
        current.mkdir(parents=True, exist_ok=True)
        self.assertEqual(get_config_dir(self.base, self.config), current)

    def test_distinct_styles_keep_independent_pipeline_state(self):
        first = get_config_dir(self.base, self.config)
        self.save_checkpoints(first, self.config)
        second = get_config_dir(self.base, self.other)
        self.assertNotEqual(first, second)
        self.assertEqual(detect_start_stage(self.base, second, self.other), "plan")
        self.save_checkpoints(second, self.other)
        first_before = (first / "state.json").read_bytes()
        with patch("paper2slides.core.pipeline.run_generate_stage", new=AsyncMock()) as generate:
            asyncio.run(run_pipeline(self.base, second, self.other, "generate"))
            generate.assert_awaited_once_with(self.base, second, self.other)
        self.assertEqual((first / "state.json").read_bytes(), first_before)
        self.assertEqual(load_state(second)["config"]["custom_style"], self.other["custom_style"])

    def test_legacy_resume_and_colliding_new_style_stay_independent(self):
        self.save_checkpoints(self.legacy, self.config)
        with patch("paper2slides.core.pipeline.run_generate_stage", new=AsyncMock()) as generate:
            asyncio.run(run_pipeline(self.base, get_config_dir(self.base, self.config), self.config, "generate"))
            generate.assert_awaited_once_with(self.base, self.legacy, self.config)
        before = (self.legacy / "state.json").read_bytes()
        second = get_config_dir(self.base, self.other)
        self.assertNotEqual(second, self.legacy)
        self.assertEqual(detect_start_stage(self.base, second, self.other), "plan")
        self.save_checkpoints(second, self.other)
        with patch("paper2slides.core.pipeline.run_generate_stage", new=AsyncMock()):
            asyncio.run(run_pipeline(self.base, second, self.other, "generate"))
        self.assertEqual((self.legacy / "state.json").read_bytes(), before)
        self.assertEqual(load_state(second)["config"]["custom_style"], self.other["custom_style"])

    def test_catalog_keeps_legacy_and_new_custom_directories(self):
        self.save_checkpoints(self.legacy, self.config)
        current = get_mode_dir(self.base, self.other) / get_config_name(self.other)
        self.save_checkpoints(current, self.other)
        with patch("paper2slides.core.pipeline.logger") as logger:
            list_outputs(self.temp.name)
        lines = [str(call.args[0]) for call in logger.info.call_args_list]
        self.assertTrue(any(f"{self.legacy.name}/" in line for line in lines))
        self.assertTrue(any(f"{current.name}/" in line for line in lines))
        self.assertEqual(sum("plan[" in line for line in lines), 2)


if __name__ == "__main__":
    unittest.main()
