import sys
from types import ModuleType

# Mock catboost before any import
catboost_mock = ModuleType("catboost")
catboost_mock.CatBoostRegressor = type("CatBoostRegressor", (object,), {})
catboost_mock.Pool = type("Pool", (object,), {})
catboost_mock.cv = lambda *args, **kwargs: None
sys.modules["catboost"] = catboost_mock

sys.path.insert(0, '/workspace/ok_avmkit/ok_avmkit')
sys.path.insert(0, '/workspace/openavmkit')

try:
    from openavmkit.modeling import run_pipeline
    print("SUCCESS: openavmkit imported successfully with mocked catboost!")
except Exception as e:
    import traceback
    traceback.print_exc()
