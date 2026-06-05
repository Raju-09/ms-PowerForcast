"""
Unit tests for PowerForecast AI Flask API endpoints.
Tests work in both 'model loaded' and 'demo mode' states.

Run with:
    cd ms-PowerForcast
    pytest tests/ -v
"""
import json
from datetime import date


# ===========================================================
# HELPERS
# ===========================================================
VALID_PREDICT_PAYLOAD = {
    "date": "2025-08-15",
    "hour": 14,
    "temperature": 32.5,
    "humidity": 55
}

VALID_COMPARE_PAYLOAD = {
    "scenario_a": {"date": "2025-06-05", "hour": 14, "temperature": 25, "humidity": 60},
    "scenario_b": {"date": "2025-06-05", "hour": 14, "temperature": 38, "humidity": 30}
}

VALID_FORECAST_PAYLOAD = {
    "date": str(date.today()),
    "hour": 0,
    "hours": 24,
    "temperature": 28,
    "humidity": 65
}

VALID_RECS_PAYLOAD = {
    "prediction": 6500,
    "hour": 19,
    "temperature": 36,
    "is_weekend": False,
    "is_holiday": False,
    "is_peak": True
}


def post_json(client, url, payload):
    return client.post(
        url,
        data=json.dumps(payload),
        content_type="application/json"
    )


# ===========================================================
# PAGE ROUTES — all must return 200
# ===========================================================
class TestPageRoutes:
    def test_home_page(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert b"PowerForecast" in res.data

    def test_predict_page(self, client):
        res = client.get("/predict")
        assert res.status_code == 200
        assert b"Predict" in res.data

    def test_forecast_page(self, client):
        res = client.get("/forecast")
        assert res.status_code == 200
        assert b"Forecast" in res.data

    def test_anomalies_page(self, client):
        res = client.get("/anomalies")
        assert res.status_code == 200
        assert b"Anomal" in res.data

    def test_dashboard_page(self, client):
        res = client.get("/dashboard")
        assert res.status_code == 200
        assert b"Dashboard" in res.data

    def test_whatif_page(self, client):
        res = client.get("/whatif")
        assert res.status_code == 200
        assert b"What-If" in res.data or b"Scenario" in res.data

    def test_status_page(self, client):
        res = client.get("/status")
        assert res.status_code == 200
        assert b"Status" in res.data

    def test_api_docs_page(self, client):
        res = client.get("/api-docs")
        assert res.status_code == 200
        assert b"API" in res.data


# ===========================================================
# /api/health
# ===========================================================
class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200

    def test_health_has_required_fields(self, client):
        res = client.get("/api/health")
        data = json.loads(res.data)
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert "model_loaded" in data
        assert "demo_mode" in data
        assert "prediction_count" in data
        assert "endpoints" in data

    def test_health_endpoints_list_not_empty(self, client):
        data = json.loads(client.get("/api/health").data)
        assert isinstance(data["endpoints"], list)
        assert len(data["endpoints"]) > 0

    def test_health_model_info_present(self, client):
        data = json.loads(client.get("/api/health").data)
        mi = data["model_info"]
        assert "name" in mi
        assert "features" in mi


# ===========================================================
# /api/predict
# ===========================================================
class TestPredictEndpoint:
    def test_valid_payload_returns_success(self, client):
        res  = post_json(client, "/api/predict", VALID_PREDICT_PAYLOAD)
        data = json.loads(res.data)
        assert res.status_code == 200
        assert data["success"] is True
        assert "prediction" in data
        assert data["unit"] == "kWh"

    def test_prediction_is_positive(self, client):
        data = json.loads(post_json(client, "/api/predict", VALID_PREDICT_PAYLOAD).data)
        assert data["prediction"] > 0

    def test_prediction_contains_input_summary(self, client):
        data = json.loads(post_json(client, "/api/predict", VALID_PREDICT_PAYLOAD).data)
        assert "input" in data
        inp = data["input"]
        assert inp["date"] == VALID_PREDICT_PAYLOAD["date"]
        assert inp["hour"] == VALID_PREDICT_PAYLOAD["hour"]

    def test_prediction_contains_model_info(self, client):
        data = json.loads(post_json(client, "/api/predict", VALID_PREDICT_PAYLOAD).data)
        assert "model_info" in data
        assert "name" in data["model_info"]

    def test_demo_mode_flag_present(self, client):
        data = json.loads(post_json(client, "/api/predict", VALID_PREDICT_PAYLOAD).data)
        assert "demo_mode" in data

    def test_missing_date_returns_400(self, client):
        payload = {"hour": 12, "temperature": 25, "humidity": 60}
        res = post_json(client, "/api/predict", payload)
        assert res.status_code == 400
        data = json.loads(res.data)
        assert data["success"] is False

    def test_empty_body_returns_400(self, client):
        res = client.post("/api/predict", data="", content_type="application/json")
        assert res.status_code == 400

    def test_holiday_detection_independence_day(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "date": "2025-08-15"}
        data = json.loads(post_json(client, "/api/predict", payload).data)
        assert data["success"] is True
        assert data["input"]["is_holiday"] is True

    def test_non_holiday_date(self, client):
        payload = {**VALID_PREDICT_PAYLOAD, "date": "2025-06-10"}
        data = json.loads(post_json(client, "/api/predict", payload).data)
        assert data["success"] is True
        # June 10 is not in the holidays list
        assert data["input"]["is_holiday"] is False

    def test_peak_hour_detected(self, client):
        # Hour 8 is a morning peak hour
        payload = {**VALID_PREDICT_PAYLOAD, "hour": 8}
        data = json.loads(post_json(client, "/api/predict", payload).data)
        assert data["success"] is True
        assert data["input"]["is_peak"] is True

    def test_offpeak_hour_detected(self, client):
        # Hour 3 is off-peak
        payload = {**VALID_PREDICT_PAYLOAD, "hour": 3}
        data = json.loads(post_json(client, "/api/predict", payload).data)
        assert data["success"] is True
        assert data["input"]["is_peak"] is False


# ===========================================================
# /api/compare (What-If)
# ===========================================================
class TestCompareEndpoint:
    def test_valid_compare_returns_success(self, client):
        res  = post_json(client, "/api/compare", VALID_COMPARE_PAYLOAD)
        data = json.loads(res.data)
        assert res.status_code == 200
        assert data["success"] is True

    def test_compare_returns_both_scenarios(self, client):
        data = json.loads(post_json(client, "/api/compare", VALID_COMPARE_PAYLOAD).data)
        assert "scenario_a" in data
        assert "scenario_b" in data
        assert "prediction" in data["scenario_a"]
        assert "prediction" in data["scenario_b"]

    def test_compare_returns_comparison_block(self, client):
        data = json.loads(post_json(client, "/api/compare", VALID_COMPARE_PAYLOAD).data)
        cmp  = data["comparison"]
        assert "difference_kwh" in cmp
        assert "difference_percent" in cmp
        assert "higher_scenario" in cmp
        assert cmp["unit"] == "kWh"

    def test_compare_difference_is_consistent(self, client):
        data = json.loads(post_json(client, "/api/compare", VALID_COMPARE_PAYLOAD).data)
        pA   = data["scenario_a"]["prediction"]
        pB   = data["scenario_b"]["prediction"]
        diff = data["comparison"]["difference_kwh"]
        assert abs(diff - (pB - pA)) < 0.01  # floating-point tolerance

    def test_compare_hot_scenario_higher_than_mild(self, client):
        # Scenario B has 38°C vs A's 25°C — B should be higher
        data = json.loads(post_json(client, "/api/compare", VALID_COMPARE_PAYLOAD).data)
        assert data["comparison"]["higher_scenario"] == "B"

    def test_compare_empty_body_returns_400(self, client):
        res = client.post("/api/compare", data="", content_type="application/json")
        assert res.status_code == 400


# ===========================================================
# /api/forecast-multi
# ===========================================================
class TestForecastMultiEndpoint:
    def test_24h_forecast_returns_24_predictions(self, client):
        res  = post_json(client, "/api/forecast-multi", VALID_FORECAST_PAYLOAD)
        data = json.loads(res.data)
        assert res.status_code == 200
        assert data["success"] is True
        assert len(data["predictions"]) == 24

    def test_48h_forecast_returns_48_predictions(self, client):
        payload = {**VALID_FORECAST_PAYLOAD, "hours": 48}
        data    = json.loads(post_json(client, "/api/forecast-multi", payload).data)
        assert data["success"] is True
        assert len(data["predictions"]) == 48

    def test_72h_forecast_returns_72_predictions(self, client):
        payload = {**VALID_FORECAST_PAYLOAD, "hours": 72}
        data    = json.loads(post_json(client, "/api/forecast-multi", payload).data)
        assert data["success"] is True
        assert len(data["predictions"]) == 72

    def test_invalid_hours_returns_400(self, client):
        payload = {**VALID_FORECAST_PAYLOAD, "hours": 100}
        res = post_json(client, "/api/forecast-multi", payload)
        assert res.status_code == 400

    def test_forecast_predictions_have_confidence_interval(self, client):
        data = json.loads(post_json(client, "/api/forecast-multi", VALID_FORECAST_PAYLOAD).data)
        for p in data["predictions"]:
            assert "lower_bound" in p
            assert "upper_bound" in p
            assert p["lower_bound"] <= p["prediction"] <= p["upper_bound"]

    def test_forecast_has_metadata(self, client):
        data = json.loads(post_json(client, "/api/forecast-multi", VALID_FORECAST_PAYLOAD).data)
        assert "metadata" in data
        assert data["forecast_hours"] == 24


# ===========================================================
# /api/recommendations
# ===========================================================
class TestRecommendationsEndpoint:
    def test_valid_payload_returns_success(self, client):
        res  = post_json(client, "/api/recommendations", VALID_RECS_PAYLOAD)
        data = json.loads(res.data)
        assert res.status_code == 200
        assert data["success"] is True

    def test_recommendations_list_present(self, client):
        data = json.loads(post_json(client, "/api/recommendations", VALID_RECS_PAYLOAD).data)
        assert "recommendations" in data
        assert isinstance(data["recommendations"], list)

    def test_each_recommendation_has_required_fields(self, client):
        data = json.loads(post_json(client, "/api/recommendations", VALID_RECS_PAYLOAD).data)
        for rec in data["recommendations"]:
            assert "title" in rec
            assert "message" in rec
            assert "priority" in rec
            assert "action" in rec

    def test_high_demand_gets_recommendation(self, client):
        payload = {**VALID_RECS_PAYLOAD, "prediction": 9000}
        data    = json.loads(post_json(client, "/api/recommendations", payload).data)
        assert data["count"] > 0


# ===========================================================
# /api/model-info
# ===========================================================
class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        assert client.get("/api/model-info").status_code == 200

    def test_returns_model_info(self, client):
        data = json.loads(client.get("/api/model-info").data)
        assert data["success"] is True
        assert "model" in data
        assert "r2_score" in data["model"]

    def test_returns_demand_statistics(self, client):
        data = json.loads(client.get("/api/model-info").data)
        assert "demand_statistics" in data


# ===========================================================
# /api/feature-importance
# ===========================================================
class TestFeatureImportanceEndpoint:
    def test_returns_200(self, client):
        assert client.get("/api/feature-importance").status_code == 200

    def test_returns_features_list(self, client):
        data = json.loads(client.get("/api/feature-importance").data)
        assert data["success"] is True
        assert "data" in data
        assert len(data["data"]) > 0

    def test_features_sorted_by_importance(self, client):
        data = json.loads(client.get("/api/feature-importance").data)
        imps = [f["importance"] for f in data["data"]]
        assert imps == sorted(imps, reverse=True)
