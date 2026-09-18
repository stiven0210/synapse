"""Self-contained demo of SYNAPSE's two-speed decision cycle end to end.

Unlike `scripts/validacion_end_to_end.py` (which needs the real, downloaded
ULB dataset), this script generates a small synthetic transaction stream on
the fly so it can run with nothing but the repo and `pip install -r
requirements.txt` -- no dataset download required.

It uses the real Calibrator-adjacent training step, the real `Bridge`
(`src/puente.py`), the real `Executor` (`src/ejecutor.py`), and the real
`CicloDecision` orchestrator (`src/ciclo.py`) -- the only simplification
versus the full Domain 1 pipeline is a smaller, synthetic dataset and a
fixed 0.5 decision threshold instead of the F1-optimal search
`src/calibrador.py` does on real data.

Run with (from the repo root, same convention as every other script here):
    python -m scripts.demo
"""
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.ciclo import CicloDecision
from src.features_recursivas import calcular_features_recursivas_batch
from src.puente import publicar

FEATURES = ["Amount", "monto_ewma_global", "conteo_ventana_global"]
N_FILAS = 300
N_DECISIONES_DEMO = 10


def generar_transacciones_sinteticas(n: int, semilla: int = 42) -> pd.DataFrame:
    """Small synthetic stream: mostly modest amounts, with a rare pattern of
    large amounts standing in for "fraud" -- just complex enough to give a
    logistic regression a real, learnable signal, with no relation to any
    real dataset."""
    rng = np.random.default_rng(semilla)
    tiempos = np.cumsum(rng.uniform(1.0, 4.0, size=n))
    es_fraude = rng.random(n) < 0.08
    montos = np.where(
        es_fraude,
        rng.uniform(1500, 5000, size=n),
        rng.lognormal(mean=3.2, sigma=0.6, size=n),
    )
    return pd.DataFrame({"Time": tiempos, "Amount": montos, "Class": es_fraude.astype(int)})


def entrenar_artefacto_demo(df_train: pd.DataFrame) -> dict:
    """Minimal training step for the demo -- same feature set and artifact
    shape as `docs/adr/0001-policy-artifact.md`, but skipping the
    F1-optimal threshold search `src/calibrador.py` does on real data, for
    simplicity."""
    modelo = LogisticRegression(class_weight="balanced")
    modelo.fit(df_train[FEATURES], df_train["Class"])
    return {
        "version": 1,
        "fecha_calibracion": datetime.now(timezone.utc).isoformat(),
        "modelo": "regresion_logistica",
        "features": FEATURES,
        "coeficientes": modelo.coef_[0].tolist(),
        "intercepto": float(modelo.intercept_[0]),
        "umbral_decision": 0.5,
        "metricas_validacion": {},
    }


def main() -> None:
    print("SYNAPSE demo -- synthetic data, real decision pipeline\n")

    df = generar_transacciones_sinteticas(N_FILAS)
    df = calcular_features_recursivas_batch(df)
    df_train, df_demo = df.iloc[:-N_DECISIONES_DEMO], df.iloc[-N_DECISIONES_DEMO:]

    print(f"Training a policy artifact on {len(df_train)} synthetic transactions...")
    artefacto = entrenar_artefacto_demo(df_train)

    with tempfile.TemporaryDirectory() as tmp:
        ruta_artefacto = Path(tmp) / "artefacto_demo.json"
        publicar(artefacto, ruta_artefacto)  # the real Bridge: atomic write, validated
        ciclo = CicloDecision(ruta_artefacto=ruta_artefacto)

        print(f"\nDeciding on {N_DECISIONES_DEMO} new transactions, one at a time:\n")
        print(f"{'#':>3}  {'Amount':>10}  {'Flagged':>8}  {'Score':>8}  {'Reason':<12}  {'us/decision':>12}")

        latencias = []
        for i, fila in enumerate(df_demo.itertuples(), start=1):
            transaccion = {"Amount": fila.Amount, "Time": fila.Time}
            inicio = time.perf_counter()
            decision = ciclo.decidir(transaccion)
            latencias.append((time.perf_counter() - inicio) * 1_000_000)

            score = f"{decision.score:.3f}" if decision.score is not None else "n/a"
            print(
                f"{i:>3}  {fila.Amount:>10.2f}  {str(decision.es_sospechosa):>8}  "
                f"{score:>8}  {decision.razon:<12}  {latencias[-1]:>12.2f}"
            )

        print(f"\nAverage measured latency: {sum(latencias) / len(latencias):.2f} us/decision")
        print(
            "\nThis is the same Bridge -> Executor -> Veto path used by the validated "
            "domains in the README, just fed synthetic data instead of a real dataset."
        )


if __name__ == "__main__":
    main()
