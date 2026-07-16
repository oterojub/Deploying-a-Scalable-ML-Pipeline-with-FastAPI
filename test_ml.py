import pytest
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
    save_model,
    load_model,
)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv("data/census.csv")
    return df


@pytest.fixture(scope="module")
def trained_model(data):
    train = data.sample(frac=0.8, random_state=42)
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)
    return model, encoder, lb


def test_train_model_returns_classifier(data):
    """Test that train_model returns a RandomForestClassifier."""
    train = data.sample(frac=0.8, random_state=42)
    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


def test_inference_output_shape(data, trained_model):
    """Test that inference returns predictions with the correct shape."""
    model, encoder, lb = trained_model
    test = data.sample(frac=0.2, random_state=42)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_test)
    assert preds.shape == y_test.shape


def test_compute_model_metrics_range(data, trained_model):
    """Test that precision, recall, and F1 are all between 0 and 1."""
    model, encoder, lb = trained_model
    test = data.sample(frac=0.2, random_state=42)
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1


def test_save_and_load_model(tmp_path, trained_model):
    """Test that a model can be saved and loaded correctly."""
    model, _, _ = trained_model
    model_path = tmp_path / "test_model.pkl"
    save_model(model, model_path)
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, RandomForestClassifier)
