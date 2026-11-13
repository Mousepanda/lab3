import numpy as np
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    ClassificationMetrics,
    RegressionInput,
    RegressionResult,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)

    # Масштабирование признаков (стандартизация)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Избегаем деления на ноль

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Инициализация весов
    n_features = X_train_scaled.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    learning_rate = data.learning_rate
    epochs = data.epochs
    m = X_train_scaled.shape[0]

    loss_history = []

    # Пакетный градиентный спуск
    for epoch in range(epochs):
        # Прямое распространение
        predictions = X_train_scaled @ weights + bias

        # MSE loss
        mse = np.mean((predictions - y_train) ** 2)
        loss_history.append(float(mse))

        # Градиенты
        error = predictions - y_train
        dw = (2 / m) * (X_train_scaled.T @ error)
        db = (2 / m) * np.sum(error)

        # Обновление параметров
        weights -= learning_rate * dw
        bias -= learning_rate * db

    # Прогнозы на тестовой выборке
    test_predictions = X_test_scaled @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        predictions=test_predictions
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=float)
    X_test = np.array(data.test_features, dtype=float)

    # Масштабирование признаков
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # Инициализация весов
    n_features = X_train_scaled.shape[1]
    weights = np.zeros(n_features)
    bias = 0.0

    learning_rate = data.learning_rate
    epochs = data.epochs
    m = X_train_scaled.shape[0]

    loss_history = []

    # Сигмоида
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Пакетный градиентный спуск
    for epoch in range(epochs):
        # Прямое распространение
        z = X_train_scaled @ weights + bias
        predictions = sigmoid(z)

        # Binary cross-entropy loss
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        bce = -np.mean(y_train * np.log(predictions) + (1 - y_train) * np.log(1 - predictions))
        loss_history.append(float(bce))

        # Градиенты
        error = predictions - y_train
        dw = (1 / m) * (X_train_scaled.T @ error)
        db = (1 / m) * np.sum(error)

        # Обновление параметров
        weights -= learning_rate * dw
        bias -= learning_rate * db

    # Прогнозы на тестовой выборке
    z_test = X_test_scaled @ weights + bias
    probabilities = sigmoid(z_test)
    predictions = (probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=bias,
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    X_train = np.array(data.train_features, dtype=float)
    y_train = np.array(data.train_target, dtype=int)
    X_test = np.array(data.test_features, dtype=float)
    y_test = np.array(data.test_target, dtype=int)

    # Обучаем собственную логистическую регрессию
    own_logistic_result = train_logistic_classifier(data)
    own_probabilities = own_logistic_result.probabilities
    own_predictions = own_logistic_result.predictions

    # Обучаем Decision Tree
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    dt_probabilities = dt_classifier.predict_proba(X_test)[:, 1]
    dt_predictions = dt_classifier.predict(X_test)

    # Обучаем KNN
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    knn_probabilities = knn_classifier.predict_proba(X_test)[:, 1]
    knn_predictions = knn_classifier.predict(X_test)

    # Функция для вычисления метрик
    def compute_metrics(y_true, y_pred, y_prob):
        return ClassificationMetrics(
            roc_auc=roc_auc_score(y_true, y_prob),
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0)
        )

    # Вычисляем метрики для всех моделей
    logistic_metrics = compute_metrics(y_test, own_predictions, own_probabilities)
    dt_metrics = compute_metrics(y_test, dt_predictions, dt_probabilities)
    knn_metrics = compute_metrics(y_test, knn_predictions, knn_probabilities)

    return BaselineComparisonResult(
        logistic_regression=logistic_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics
    )
