import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

from state_encoder import encode_observation, encode_engineered, FEATURE_NAMES_OBS, FEATURE_NAMES_ENG
from data_loader import parse_logs

# === CONSTANTES ===
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT_DIR, "data", "pluribus_logs", "*.txt")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
ACCIONES = ['fold', 'check', 'call', 'bet', 'raise']
STREETS = ['Preflop', 'Flop', 'Turn', 'River']
RANDOM_STATE = 42


def save_figure(filename):
    """Guarda la figura actual; si la imagen está abierta en otro programa (p. ej. Fotos) y
    Windows la bloquea, reintenta y, si no puede, avisa en vez de interrumpir la ejecución."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    for _ in range(3):
        try:
            plt.savefig(path, dpi=300)
            break
        except OSError:
            time.sleep(1)
    else:
        print(f"AVISO: no se pudo guardar {filename} (¿está abierta en otro programa?)")
    plt.close()


def build_model():
    # Red pequeña y regularizada: ~20 variables de entrada y ~12k ejemplos de entrenamiento
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                      max_iter=500, random_state=RANDOM_STATE),
    )


def metrics(y_true, y_pred, proba=None, classes=None):
    m = {'accuracy': accuracy_score(y_true, y_pred),
         'balanced_acc': balanced_accuracy_score(y_true, y_pred),
         'macro_F1': f1_score(y_true, y_pred, average='macro')}
    if proba is not None:
        m['log_loss'] = log_loss(y_true, proba, labels=classes)
    return m


def print_metrics(name, m):
    print(f"{name:<32} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))


def run_model(name, X, y, groups, train_idx, test_idx, street_test, feature_names):
    """Validación cruzada, entrenamiento, evaluación e importancia por permutación de un modelo."""
    print(f"\n{'=' * 20} MODELO {name} ({X.shape[1]} variables) {'=' * 20}")
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    print("Realizando validación cruzada (5-fold agrupado por sesión)...")
    cv = cross_validate(build_model(), X_train, y_train, groups=groups[train_idx], cv=GroupKFold(n_splits=5),
                        scoring=['accuracy', 'f1_macro', 'neg_log_loss'])
    print(f"Accuracy CV: {cv['test_accuracy'].mean():.3f} ± {cv['test_accuracy'].std():.3f}")
    print(f"Macro F1 CV: {cv['test_f1_macro'].mean():.3f} ± {cv['test_f1_macro'].std():.3f}")
    print(f"Log loss CV: {-cv['test_neg_log_loss'].mean():.3f} ± {cv['test_neg_log_loss'].std():.3f}")

    print("Entrenando red neuronal...")
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    print("\n--- Resultados en test ---")
    print_metrics("Train (referencia sobreajuste)", metrics(y_train, model.predict(X_train)))
    test_metrics = metrics(y_test, y_pred, proba, model.classes_)
    print_metrics("Test", test_metrics)
    print(classification_report(y_test, y_pred, labels=ACCIONES, zero_division=0))
    print("Por calle:")
    for s, street in enumerate(STREETS):
        mask = street_test == s
        print_metrics(f"  {street} (n={mask.sum()})", metrics(y_test[mask], y_pred[mask]))
        test_metrics[f'acc_{street}'] = accuracy_score(y_test[mask], y_pred[mask])

    # Matriz de confusión (normalizada por fila: qué predice el modelo para cada acción real)
    cm = confusion_matrix(y_test, y_pred, labels=ACCIONES, normalize='true')
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ACCIONES).plot(
        cmap='Blues', xticks_rotation=45, values_format='.2f')
    plt.title(f'Matriz de confusión — modelo {name}')
    plt.tight_layout()
    save_figure(f"confusion_matrix_{name}.png")

    # Importancia por permutación: cuánto empeora el log loss en test al desordenar cada variable.
    # Ojo: con variables correlacionadas (p. ej. hand_category y pair_strength) la importancia se reparte.
    print("Calculando importancia por permutación...")
    pi = permutation_importance(model, X_test, y_test, scoring='neg_log_loss', n_repeats=10,
                                random_state=RANDOM_STATE)
    importance = pd.DataFrame({'mean': pi.importances_mean, 'std': pi.importances_std},
                              index=feature_names).sort_values('mean', ascending=False)
    print(importance.round(4).to_string())

    return test_metrics, importance


def plot_importances(importances):
    fig, axes = plt.subplots(1, len(importances), figsize=(6 * len(importances), 7))
    for ax, (name, imp) in zip(axes, importances.items()):
        imp = imp.iloc[::-1]
        ax.barh(imp.index, imp['mean'], xerr=imp['std'], color='steelblue')
        ax.set_title(f'Modelo {name}')
        ax.set_xlabel('Aumento del log loss al permutar')
    plt.tight_layout()
    save_figure("permutation_importance.png")


def main():
    # === 1. Cargar los datos (estado ANTES de cada acción de Pluribus) ===
    print("Leyendo logs de Pluribus...")
    df = parse_logs(LOG_PATH)
    df = df[df['action'].isin(ACCIONES)].reset_index(drop=True)
    print(f"Total decisiones de Pluribus: {len(df)} en {df['session'].nunique()} sesiones")
    print("Distribución de acciones:\n", df['action'].value_counts(normalize=True).round(3).to_string())

    # === 2. Codificar X e y para los dos modelos ===
    print("Codificando jugadas...")
    X_obs = np.vstack(df.apply(encode_observation, axis=1).values)
    X_eng = np.vstack(df.apply(encode_engineered, axis=1).values)
    y = df['action'].values
    groups = df['session'].values

    print("\n=== EJEMPLO DE MANO CODIFICADA ===")
    i = int(np.argmax(df['street'].values == 1))  # primera decisión en el flop
    r = df.iloc[i]
    print(f"{r['hole1']} {r['hole2']} | board {r['board']} | {STREETS[r['street']]} -> {y[i]}")
    for names, X, label in [(FEATURE_NAMES_OBS, X_obs, 'A'), (FEATURE_NAMES_ENG, X_eng, 'C')]:
        print(f" Modelo {label}: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(names, X[i])))

    # === 3. Separar train/test por SESIÓN (el mismo split para los dos modelos) ===
    # Las decisiones de una misma mano/sesión están correlacionadas; si se reparten entre train
    # y test el modelo se evalúa con situaciones casi idénticas a las que ya ha visto.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X_obs, y, groups))
    y_train, y_test = y[train_idx], y[test_idx]
    street_test = df['street'].values[test_idx]
    print(f"\nTrain: {len(train_idx)} decisiones ({len(set(groups[train_idx]))} sesiones) | "
          f"Test: {len(test_idx)} decisiones ({len(set(groups[test_idx]))} sesiones)")

    # === 4. Baselines ===
    print("\n=== BASELINES ===")
    majority = pd.Series(y_train).value_counts().idxmax()
    print_metrics("Clase mayoritaria", metrics(y_test, np.full(len(y_test), majority)))
    street_majority = pd.Series(y_train).groupby(df['street'].values[train_idx]).agg(lambda s: s.value_counts().idxmax())
    print_metrics("Mayoritaria por calle", metrics(y_test, pd.Series(street_test).map(street_majority).values))

    # === 5. Modelos A (observación, referencia) y C (feature engineering, definitivo) ===
    # El modelo B (C + equity Monte Carlo) se descartó en v4: no mejoraba a C (ver docs/CODIFICACION.md, §7).
    results, importances = {}, {}
    for name, X, feature_names in [('A', X_obs, FEATURE_NAMES_OBS), ('C', X_eng, FEATURE_NAMES_ENG)]:
        results[name], importances[name] = run_model(name, X, y, groups, train_idx, test_idx,
                                                     street_test, feature_names)
    plot_importances(importances)

    print("\n=== COMPARACIÓN A (observación) vs C (feature engineering) ===")
    print(pd.DataFrame(results).round(3).to_string())


if __name__ == "__main__":
    main()
