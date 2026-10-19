"""Path regressions, runnable without installing model or parser dependencies."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


# core.__init__ eagerly imports the model pipeline; load this pure module alone.
spec = importlib.util.spec_from_file_location(
    "paper2slides_paths", Path(__file__).resolve().parents[1] / "paper2slides/core/paths.py"
)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)


class ConfigPathTests(unittest.TestCase):
    def test_distinct_custom_styles_do_not_reuse_plan_checkpoint(self):
        for first, second in (
            ("Watercolor style with blue backgrounds", "Watercolor style with red backgrounds"),
            ("blue green", "blue_green"),
            ("blue/green", "blue_green"),
        ):
            for output_type in ("slides", "poster"):
                with self.subTest(first=first, second=second, output_type=output_type):
                    with tempfile.TemporaryDirectory() as directory:
                        config = {"style": "custom", "custom_style": first,
                                  "output_type": output_type}
                        base = Path(directory)
                        original = paths.get_config_dir(base, config)
                        original.mkdir(parents=True)
                        paths.get_plan_checkpoint(original).write_text("{}")
                        changed = paths.get_config_dir(base, {**config, "custom_style": second})
                        self.assertFalse(paths.get_plan_checkpoint(changed).exists())
                        self.assertTrue(paths.get_plan_checkpoint(original).exists())

    def test_same_custom_style_reuses_directory(self):
        config = {"style": "custom", "custom_style": "水彩画 / blue"}
        self.assertEqual(paths.get_config_name(config), paths.get_config_name(dict(config)))
        self.assertNotIn("/", paths.get_config_name(config))
        self.assertTrue(paths.get_config_name(config).isascii())

    def test_predefined_names_unchanged(self):
        self.assertEqual(paths.get_config_name({}), "slides_academic_medium")
        self.assertEqual(paths.get_config_name({"style": "doraemon", "slides_length": "long"}),
                         "slides_doraemon_long")
        self.assertEqual(paths.get_config_name({"output_type": "poster", "poster_density": "dense"}),
                         "poster_academic_dense")

    def test_empty_custom_style_unchanged(self):
        for custom in (None, ""):
            self.assertEqual(paths.get_config_name({"style": "custom", "custom_style": custom}),
                             "slides_custom_custom_medium")


if __name__ == "__main__":
    unittest.main()
