import importlib


def test_import_sex_config():
    import FIT_python.pipeline_sex.sex_config as sex_config
    importlib.reload(sex_config)

