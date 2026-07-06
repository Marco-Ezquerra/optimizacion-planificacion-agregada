import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

try:
    import folium
except Exception:
    folium = None


@dataclass
class SolverConfig:
    name: str
    time_limit: int
    msg: bool


def choose_solver(preferred: str, time_limit: int, msg: bool) -> SolverConfig:
    preferred = preferred.lower()

    if preferred in {"auto", "gurobi"}:
        try:
            _ = pulp.GUROBI(timeLimit=1, msg=False)
            return SolverConfig(name="gurobi", time_limit=time_limit, msg=msg)
        except Exception:
            if preferred == "gurobi":
                raise RuntimeError("Gurobi no esta disponible. Usa --solver auto o --solver cbc.")

    if preferred in {"auto", "cbc"}:
        return SolverConfig(name="cbc", time_limit=time_limit, msg=msg)

    raise ValueError("Solver no valido. Opciones: auto, gurobi, cbc")


def build_solver_instance(config: SolverConfig):
    if config.name == "gurobi":
        return pulp.GUROBI(timeLimit=config.time_limit, msg=config.msg)
    return pulp.PULP_CBC_CMD(timeLimit=config.time_limit, msg=config.msg)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"ID", "Latitud", "Longitud", "Peso"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {sorted(missing)}")
    return df.copy()


def initialize_centers(coords: np.ndarray, n_techs: int, random_state: int, n_init: int) -> np.ndarray:
    model = KMeans(n_clusters=n_techs, random_state=random_state, n_init=n_init)
    model.fit(coords)
    return model.cluster_centers_.copy()


def solve_assignment_ala(
    coords: np.ndarray,
    weights: np.ndarray,
    init_centers: np.ndarray,
    lambda_value: float,
    max_iterations: int,
    balance_ratio: float,
    solver_cfg: SolverConfig,
) -> Dict:
    n_installations = coords.shape[0]
    n_techs = init_centers.shape[0]

    centers = init_centers.copy()
    dist_initial = cdist(coords, centers, metric="euclidean")
    d_ref = float(dist_initial.min(axis=1).sum())
    e_ref = float(weights.sum() / n_techs)
    epsilon_hours = e_ref * balance_ratio

    best_obj = float("inf")
    best_assignment = np.zeros(n_installations, dtype=int)
    best_distance = float("inf")
    best_imbalance = float("inf")

    history = {
        "objective": [],
        "distance": [],
        "imbalance": [],
        "status": [],
    }

    for _ in range(max_iterations):
        distances = cdist(coords, centers, metric="euclidean")

        prob = pulp.LpProblem("assignment_ala", pulp.LpMinimize)
        x = pulp.LpVariable.dicts(
            "x",
            ((i, c) for i in range(n_installations) for c in range(n_techs)),
            cat="Binary",
        )
        l_max = pulp.LpVariable("L_max", lowBound=0)
        l_min = pulp.LpVariable("L_min", lowBound=0)

        sum_distance = pulp.lpSum(
            distances[i, c] * x[i, c]
            for i in range(n_installations)
            for c in range(n_techs)
        )
        load_diff = l_max - l_min

        prob += (
            lambda_value * (sum_distance / d_ref)
            + (1.0 - lambda_value) * (load_diff / e_ref)
        )

        for i in range(n_installations):
            prob += pulp.lpSum(x[i, c] for c in range(n_techs)) == 1

        for c in range(n_techs):
            load_c = pulp.lpSum(weights[i] * x[i, c] for i in range(n_installations))
            prob += load_c <= l_max
            prob += load_c >= l_min

        prob += (l_max - l_min) <= epsilon_hours

        solver = build_solver_instance(solver_cfg)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        has_solution = pulp.value(l_max) is not None
        history["status"].append(status)

        if status not in {"Optimal", "Feasible"} and not has_solution:
            break

        objective = float(pulp.value(prob.objective))
        lmax_value = float(pulp.value(l_max))
        lmin_value = float(pulp.value(l_min))
        imbalance = lmax_value - lmin_value

        assignment = np.zeros(n_installations, dtype=int)
        for i in range(n_installations):
            for c in range(n_techs):
                value = pulp.value(x[i, c])
                if value is not None and value > 0.5:
                    assignment[i] = c + 1
                    break

        distance_total = float(
            sum(
                distances[i, assignment[i] - 1]
                for i in range(n_installations)
                if assignment[i] > 0
            )
        )

        history["objective"].append(objective)
        history["distance"].append(distance_total)
        history["imbalance"].append(imbalance)

        if objective < best_obj:
            best_obj = objective
            best_assignment = assignment.copy()
            best_distance = distance_total
            best_imbalance = imbalance

        updated_centers = np.zeros_like(centers)
        for c in range(n_techs):
            members = np.where(assignment == c + 1)[0]
            if len(members) > 0:
                updated_centers[c] = coords[members].mean(axis=0)
            else:
                updated_centers[c] = centers[c]
        centers = updated_centers

    feasible = np.any(best_assignment > 0)

    return {
        "feasible": bool(feasible),
        "best_objective": best_obj if feasible else None,
        "best_assignment": best_assignment,
        "best_distance": best_distance if feasible else None,
        "best_imbalance": best_imbalance if feasible else None,
        "history": history,
        "final_centers": centers,
    }


def choose_compromise_solution(metrics_df: pd.DataFrame) -> pd.Series:
    df = metrics_df.copy()
    d_min = df["Distancia"].min()
    b_min = df["Diferencia_Horas"].min()

    df["score_compromiso"] = 0.5 * (df["Distancia"] / d_min) + 0.5 * (df["Diferencia_Horas"] / b_min)
    return df.sort_values("score_compromiso", ascending=True).iloc[0]


def build_balance_summary(solution_df: pd.DataFrame) -> pd.DataFrame:
    return (
        solution_df.groupby("ID_Tecnico")
        .agg(Num_Instalaciones=("ID", "count"), Carga_Horas=("Peso", "sum"))
        .reset_index()
        .sort_values("ID_Tecnico")
    )


def save_balance_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    mean_hours = float(summary_df["Carga_Horas"].mean())
    max_hours = float(summary_df["Carga_Horas"].max())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        summary_df["ID_Tecnico"].astype(str),
        summary_df["Carga_Horas"],
        color="#8ecae6",
        edgecolor="#1d3557",
    )
    plt.axhline(y=mean_hours, color="#d90429", linestyle="--", linewidth=2, label=f"Media: {mean_hours:.1f} h")

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.01 * max_hours,
            f"{yval:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.title("Balance de carga por tecnico")
    plt.xlabel("ID del tecnico")
    plt.ylabel("Carga total (horas)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_solution_scatter_map(solution_df: pd.DataFrame, output_path: Path) -> None:
    n_techs = int(solution_df["ID_Tecnico"].nunique())
    cmap = cm.get_cmap("tab20", n_techs)

    plt.figure(figsize=(10, 8))
    for tech in sorted(solution_df["ID_Tecnico"].unique()):
        tech_data = solution_df[solution_df["ID_Tecnico"] == tech]
        color = cmap(int(tech) - 1)
        plt.scatter(
            tech_data["Longitud"],
            tech_data["Latitud"],
            s=10,
            color=color,
            alpha=0.8,
            label=f"Tec {int(tech)}",
        )

    plt.title("Mapa de la solucion (asignacion por tecnico)")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_solution_folium_map(solution_df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    if folium is None:
        return None

    n_techs = int(solution_df["ID_Tecnico"].nunique())
    cmap = cm.get_cmap("tab20", n_techs)
    colors_hex = [mcolors.to_hex(cmap(i)) for i in range(n_techs)]

    fmap = folium.Map(
        location=[solution_df.Latitud.mean(), solution_df.Longitud.mean()],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    for _, row in solution_df.iterrows():
        tech_id = int(row["ID_Tecnico"])
        color = colors_hex[tech_id - 1]
        folium.CircleMarker(
            location=[row["Latitud"], row["Longitud"]],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"Tecnico: {tech_id}<br>ID: {int(row['ID'])}<br>Peso: {row['Peso']}",
        ).add_to(fmap)

    centroids = (
        solution_df.groupby("ID_Tecnico")[["Latitud", "Longitud"]]
        .mean()
        .reset_index()
    )
    for _, c in centroids.iterrows():
        folium.Marker(
            location=[c["Latitud"], c["Longitud"]],
            icon=folium.Icon(color="black", icon="star"),
            popup=f"Centroide tecnico {int(c['ID_Tecnico'])}",
        ).add_to(fmap)

    fmap.save(str(output_path))
    return output_path


def save_winner_iteration_charts(history: Dict[str, List[float]], output_path: Path) -> None:
    if len(history.get("objective", [])) == 0:
        return

    xs = range(1, len(history["objective"]) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(xs, history["objective"], marker="o", color="#6a4c93", linewidth=2)
    ax1.set_title("Evolucion objetivo")
    ax1.set_xlabel("Iteracion")
    ax1.set_ylabel("Z")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(xs, history["distance"], marker="s", color="#1982c4", linewidth=2)
    ax2.set_title("Evolucion distancia")
    ax2.set_xlabel("Iteracion")
    ax2.set_ylabel("Distancia")
    ax2.grid(True, linestyle="--", alpha=0.6)

    ax3.plot(xs, history["imbalance"], marker="^", color="#ff7f11", linewidth=2)
    ax3.set_title("Evolucion balance")
    ax3.set_xlabel("Iteracion")
    ax3.set_ylabel("L_max - L_min")
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    coords = df[["Latitud", "Longitud"]].to_numpy()
    weights = df["Peso"].to_numpy()

    solver_cfg = choose_solver(args.solver, args.time_limit, args.solver_log)
    init_centers = initialize_centers(coords, args.technicians, args.random_state, args.kmeans_n_init)

    lambda_values = [float(v.strip()) for v in args.lambdas.split(",") if v.strip()]

    pareto_rows: List[Dict] = []
    assignment_columns: Dict[str, np.ndarray] = {}
    results_by_lambda: Dict[float, Dict] = {}

    for lambda_value in lambda_values:
        result = solve_assignment_ala(
            coords=coords,
            weights=weights,
            init_centers=init_centers,
            lambda_value=lambda_value,
            max_iterations=args.max_iterations,
            balance_ratio=args.balance_ratio,
            solver_cfg=solver_cfg,
        )

        if not result["feasible"]:
            continue

        results_by_lambda[round(lambda_value, 3)] = result

        pareto_rows.append(
            {
                "Lambda": lambda_value,
                "Distancia": result["best_distance"],
                "Diferencia_Horas": result["best_imbalance"],
                "Objetivo": result["best_objective"],
            }
        )
        assignment_columns[f"ID_Tecnico_L_{lambda_value:.1f}"] = result["best_assignment"]

    if not pareto_rows:
        raise RuntimeError("No se encontro ninguna solucion factible. Ajusta balance_ratio o time_limit.")

    df_metrics = pd.DataFrame(pareto_rows).sort_values("Lambda").reset_index(drop=True)
    metrics_path = out_dir / "metricas_pareto.csv"
    df_metrics.to_csv(metrics_path, index=False)

    df_master = df[["ID", "Latitud", "Longitud", "Peso"]].copy()
    for col_name, values in assignment_columns.items():
        df_master[col_name] = values
    master_path = out_dir / "soluciones_maestro_pareto.csv"
    df_master.to_csv(master_path, index=False)

    winner = choose_compromise_solution(df_metrics)
    winner_lambda = float(winner["Lambda"])
    winner_col = f"ID_Tecnico_L_{winner_lambda:.1f}"

    df_solution = df[["ID", "Latitud", "Longitud", "Peso"]].copy()
    df_solution["ID_Tecnico"] = df_master[winner_col].astype(int)
    solution_path = out_dir / "solucion.csv"
    df_solution.to_csv(solution_path, index=False)

    summary_df = build_balance_summary(df_solution)
    summary_path = out_dir / "resumen_balance.csv"
    summary_df.to_csv(summary_path, index=False)

    balance_plot_path = out_dir / "balance_carga_tecnicos.png"
    save_balance_chart(summary_df, balance_plot_path)

    map_scatter_path = out_dir / "mapa_solucion.png"
    save_solution_scatter_map(df_solution, map_scatter_path)

    map_html_path = out_dir / "mapa_solucion.html"
    folium_result = save_solution_folium_map(df_solution, map_html_path)

    winner_history_plot = out_dir / "evolucion_lambda_ganador.png"
    winner_result = results_by_lambda.get(round(winner_lambda, 3))
    if winner_result is not None:
        save_winner_iteration_charts(winner_result.get("history", {}), winner_history_plot)

    plt.figure(figsize=(10, 6))
    plt.plot(
        df_metrics["Diferencia_Horas"],
        df_metrics["Distancia"],
        marker="o",
        linestyle="-",
        linewidth=2,
    )
    for _, row in df_metrics.iterrows():
        plt.annotate(
            f"lambda={row['Lambda']:.1f}",
            (row["Diferencia_Horas"], row["Distancia"]),
            textcoords="offset points",
            xytext=(8, 5),
            ha="left",
            fontsize=9,
        )

    plt.title("Frente de Pareto: distancia vs equilibrio")
    plt.xlabel("Desequilibrio maximo de carga (horas)")
    plt.ylabel("Suma total de distancias")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    pareto_plot_path = out_dir / "frente_pareto.png"
    plt.savefig(pareto_plot_path, dpi=300)
    plt.close()

    print("Resumen de ejecucion")
    print(f"- Solver usado: {solver_cfg.name}")
    print(f"- Solucion elegida (compromiso): lambda={winner_lambda:.1f}")
    print(f"- Distancia: {winner['Distancia']:.2f}")
    print(f"- Diferencia de carga: {winner['Diferencia_Horas']:.2f} h")
    print(f"- CSV solucion: {solution_path}")
    print(f"- CSV pareto:   {metrics_path}")
    print(f"- CSV maestro:  {master_path}")
    print(f"- CSV balance:  {summary_path}")
    print(f"- Grafica:      {pareto_plot_path}")
    print(f"- Grafica bal.: {balance_plot_path}")
    print(f"- Grafica evo.: {winner_history_plot}")
    print(f"- Mapa PNG:     {map_scatter_path}")
    if folium_result is not None:
        print(f"- Mapa HTML:    {map_html_path}")
    else:
        print("- Mapa HTML:    no generado (folium no disponible)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Planificacion de instalaciones con enfoque hibrido K-Means + MILP (PuLP)."
    )
    parser.add_argument("--data", default="instalaciones.csv", help="Ruta del CSV de entrada.")
    parser.add_argument("--output-dir", default=".", help="Directorio de salida para resultados.")
    parser.add_argument("--technicians", type=int, default=20, help="Numero de tecnicos.")
    parser.add_argument(
        "--lambdas",
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Lista de lambdas separada por comas.",
    )
    parser.add_argument("--max-iterations", type=int, default=3, help="Iteraciones maximas del bucle ALA.")
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=0.15,
        help="Tolerancia de equilibrio como porcentaje de la carga media (epsilon = ratio * E_ref).",
    )
    parser.add_argument("--time-limit", type=int, default=60, help="Time limit por iteracion y lambda (segundos).")
    parser.add_argument("--solver", choices=["auto", "gurobi", "cbc"], default="auto")
    parser.add_argument("--solver-log", action="store_true", help="Muestra logs del solver.")
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_pipeline(cli_args)
