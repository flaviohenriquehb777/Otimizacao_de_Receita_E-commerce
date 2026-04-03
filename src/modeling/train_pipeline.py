import os
import json
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# SHAP opcional (evita dependências nativas em runners mínimos)
try:
    import shap  # type: ignore
    HAS_SHAP = True
except Exception:
    shap = None  # type: ignore
    HAS_SHAP = False

import mlflow

try:
    import dagshub
except Exception:
    dagshub = None

from src.config.paths import DADOS_AMOR_A_CAKES, MODELS_DIR, PROJECT_ROOT


@dataclass
class ModelSpec:
    name: str
    estimator: object
    params: dict


def init_tracking():
    """Inicializa MLflow e opcionalmente DagsHub, lendo variáveis de ambiente.

    Suporta ambos padrões de env:
    - DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME
    - DAGSHUB_OWNER / DAGSHUB_REPO
    E MLFLOW_TRACKING_URI explícito quando fornecido.
    """
    # Primeiro, configurar tracking URI (somente se houver credenciais); caso contrário, usar tracking local
    owner = os.getenv("DAGSHUB_REPO_OWNER") or os.getenv("DAGSHUB_OWNER")
    repo = os.getenv("DAGSHUB_REPO_NAME") or os.getenv("DAGSHUB_REPO")
    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    username = os.getenv("MLFLOW_TRACKING_USERNAME") or os.getenv("DAGSHUB_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")
    token_env = os.getenv("MLFLOW_TRACKING_TOKEN") or os.getenv("DAGSHUB_TOKEN")

    # Aceitar token-only (sem username/password) para MLflow em DagsHub
    has_credentials = bool(token_env) or (bool(username) and bool(password))

    def set_local_tracking():
        try:
            local_uri = (Path.cwd() / "mlruns").resolve().as_uri()
            mlflow.set_tracking_uri(local_uri)
            return True
        except Exception as e_local:
            warnings.warn(f"Falha ao definir tracking local do MLflow: {e_local}")
            return False

    if has_credentials:
        if not username and owner:
            username = owner
        # Garantir que MLflow leia credenciais via env
        if username:
            os.environ.setdefault("MLFLOW_TRACKING_USERNAME", username)
        if password:
            os.environ.setdefault("MLFLOW_TRACKING_PASSWORD", password)
        if token_env:
            os.environ.setdefault("MLFLOW_TRACKING_TOKEN", token_env)

        # Definir URI remoto (env explícito tem prioridade; senão, montar via owner/repo)
        try:
            if tracking_uri_env:
                mlflow.set_tracking_uri(tracking_uri_env)
            elif owner and repo:
                mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{repo}.mlflow")
            # Inicialização explícita do DagsHub quando em CI ou quando habilitado
            try:
                import dagshub  # type: ignore
                if owner and repo and (os.getenv("CI") or os.getenv("ENABLE_REMOTE_MLFLOW") == "1"):
                    dagshub.init(repo, owner, mlflow=True)
            except Exception as e_dag:
                warnings.warn(f"Falha ao inicializar DagsHub (init): {e_dag}. Prosseguindo com MLflow URI configurado.")
        except Exception as e_uri:
            warnings.warn(f"Falha ao definir tracking remoto do MLflow: {e_uri}. Alternando para tracking local.")
            set_local_tracking()
    else:
        # Fallback robusto: tracking local em arquivo, evitando 401/403 em ambientes sem credenciais
        set_local_tracking()

    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "max_receita_cafeterias")
    try:
        mlflow.set_experiment(exp_name)
    except Exception as e_exp:
        warnings.warn(f"Falha ao definir experimento '{exp_name}' no MLflow: {e_exp}. Alternando para tracking local e experimento padrão.")
        # Força fallback local se a configuração de experimento remoto falhar (ex.: 401)
        if set_local_tracking():
            try:
                mlflow.set_experiment(exp_name)
            except Exception as e_exp2:
                warnings.warn(f"Falha ao definir experimento local '{exp_name}': {e_exp2}. Prosseguindo com experimento padrão.")
    try:
        uri = mlflow.get_tracking_uri()
        print(f"[MLflow] Tracking URI ativo: {uri}")
        print(f"[MLflow] Experimento base: {exp_name}")
    except Exception:
        pass

    # Não chamar dagshub.init aqui para evitar fluxos OAuth interativos na CI.


def load_dataset() -> pd.DataFrame:
    """
    Carrega o dataset principal; se não existir (CI/ambiente limpo), gera um dataset sintético
    determinístico para permitir treino e testes estáveis.
    """
    if DADOS_AMOR_A_CAKES.exists():
        df = pd.read_excel(DADOS_AMOR_A_CAKES)
    else:
        rng = np.random.default_rng(42)
        n = 600
        dates = pd.date_range("2021-07-01", periods=n, freq="D")
        preco_original = rng.uniform(20.0, 60.0, size=n)
        desconto_pct = rng.uniform(0.0, 0.04, size=n)
        custo_producao = rng.uniform(5.0, 15.0, size=n)
        preco_final = preco_original * (1.0 - desconto_pct)
        # Relação: demanda cai com preço, sobe com desconto; ruído controlado
        base_qty = 500 - 3.5 * preco_final + 1200 * desconto_pct
        qty_noise = rng.normal(0.0, 25.0, size=n)
        quantidade_vendida_mes = np.clip(base_qty + qty_noise, a_min=0, a_max=None)
        df = pd.DataFrame({
            'data': dates,
            'custo_producao': custo_producao,
            'preco_original': preco_original,
            'desconto_pct': desconto_pct,
            'preco_final': preco_final,
            'quantidade_vendida_mes': quantidade_vendida_mes,
        })

    df = df.drop_duplicates()
    if 'data' in df.columns:
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
    if {'preco_final','quantidade_vendida_dia'}.issubset(df.columns):
        df['receita_dia'] = df['preco_final'] * df['quantidade_vendida_dia']
    if {'preco_final','quantidade_vendida_mes'}.issubset(df.columns):
        df['receita_mes'] = df['preco_final'] * df['quantidade_vendida_mes']
    return df


def select_features(df: pd.DataFrame):
    # Prioriza receita como alvo quando disponível; caso contrário, quantidade
    if 'receita_mes' in df.columns:
        target = 'receita_mes'
    elif 'receita_dia' in df.columns:
        target = 'receita_dia'
    else:
        target = 'quantidade_vendida_mes' if 'quantidade_vendida_mes' in df.columns else 'quantidade_vendida_dia'

    feature_cols = [c for c in ['custo_producao','preco_original','desconto_pct','preco_final'] if c in df.columns]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y, feature_cols, target


def get_model_specs(feature_cols=None, target: str | None = None):
    specs = [
        ModelSpec("LinearRegression", LinearRegression(), {}),
        ModelSpec("RidgeCV", Pipeline([('scaler', StandardScaler()), ('model', RidgeCV(alphas=np.logspace(-3, 3, 21), cv=5))]), {}),
        ModelSpec("LassoCV", Pipeline([('scaler', StandardScaler()), ('model', LassoCV(alphas=None, cv=5, max_iter=10000, tol=1e-4))]), {}),
        ModelSpec("ElasticNetCV", Pipeline([('scaler', StandardScaler()), ('model', ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9], cv=5, max_iter=10000, tol=1e-4))]), {}),
        ModelSpec("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42), {}),
        ModelSpec("GradientBoosting", GradientBoostingRegressor(random_state=42), {}),
    ]
    if XGBRegressor is not None:
        # Mapeia restrições monotônicas conforme ordem das features
        # Para alvo receita, desativamos restrições (neutras). Para quantidade, mantemos.
        is_revenue_target = target in { 'receita_mes', 'receita_dia' }
        constraints_map = {
            'custo_producao': 0 if is_revenue_target else -1,
            'preco_original': 0 if is_revenue_target else -1,
            'desconto_pct': 0 if is_revenue_target else +1,
            'preco_final': 0 if is_revenue_target else -1,
        }
        constraints = []
        if feature_cols is None:
            feature_cols = [c for c in ['custo_producao','preco_original','desconto_pct','preco_final']]
        for c in feature_cols:
            constraints.append(constraints_map.get(c, 0))
        # XGBoost aceita string com parênteses, ex.: "(0,-1,1,-1)"
        constraints_str = "(" + ",".join(str(v) for v in constraints) + ")"
        specs.append(ModelSpec(
            "XGBRegressor",
            XGBRegressor(
                random_state=42,
                n_estimators=600,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_alpha=0.0,
                reg_lambda=2.0,
                min_child_weight=5,
                gamma=0.0,
                tree_method='hist',
                objective='reg:squarederror',
                monotone_constraints=constraints_str,
            ),
            {'monotone_constraints': constraints_str}
        ))
    return specs


def _simulate_expected_profit(model, X_valid: pd.DataFrame, target: str | None = None) -> float:
    """Simula lucro esperado por linha buscando o melhor desconto até 4%.
    Compatível com alvo em quantidade ou receita.
    Retorna a média do lucro máximo por linha na validação.
    """
    if not {'preco_original','custo_producao'}.issubset(X_valid.columns):
        return float('nan')
    grid = np.arange(0.0, 0.0400001, 0.001)
    best_profit = np.full(shape=(len(X_valid),), fill_value=-np.inf, dtype=float)
    for d in grid:
        Xd = X_valid.copy()
        if 'desconto_pct' in Xd.columns:
            Xd['desconto_pct'] = d
        if 'preco_final' in Xd.columns:
            Xd['preco_final'] = X_valid['preco_original'] * (1.0 - d)
        preds = np.asarray(model.predict(Xd), dtype=float)
        preco_final = X_valid['preco_original'] * (1.0 - d)
        if target in {'receita_mes', 'receita_dia'}:
            # Lucro = Receita - Custo total; quantidade = receita / preço
            # profit = preds - custo_producao * (preds / preco_final)
            # = preds * (1 - custo_producao / preco_final)
            fator = (1.0 - (X_valid['custo_producao'] / preco_final))
            profit_d = preds * np.asarray(fator, dtype=float)
        else:
            # Alvo em quantidade: lucro por unidade * quantidade
            profit_d = (preco_final - X_valid['custo_producao']) * preds
        best_profit = np.maximum(best_profit, profit_d.values if hasattr(profit_d, 'values') else profit_d)
    return float(np.nanmean(best_profit))

def _simulate_expected_profit_holdout(model, X_valid: pd.DataFrame, target: str | None = None) -> float:
    """Calcula lucro esperado em holdout usando melhor desconto por linha (0%–4%).
    Compatível com alvo em quantidade ou receita.
    """
    try:
        return _simulate_expected_profit(model, X_valid, target)
    except Exception:
        return float('nan')


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv, target: str | None = None):
    rmses, maes, r2s, profits = [], [], [], []
    for train_idx, valid_idx in cv.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)
        # Compatibilidade com versões mais antigas do scikit-learn onde 'squared' não existe
        try:
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
        except TypeError:
            rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
        rmses.append(rmse)
        maes.append(mean_absolute_error(y_valid, y_pred))
        r2s.append(r2_score(y_valid, y_pred))
        profits.append(_simulate_expected_profit(model, X_valid, target))
    return {
        'cv_rmse_mean': float(np.mean(rmses)), 'cv_rmse_std': float(np.std(rmses)),
        'cv_mae_mean': float(np.mean(maes)), 'cv_mae_std': float(np.std(maes)),
        'cv_r2_mean': float(np.mean(r2s)), 'cv_r2_std': float(np.std(r2s)),
        'cv_expected_profit_mean': float(np.mean(profits)), 'cv_expected_profit_std': float(np.std(profits)),
    }


def export_linear_json(model, feature_cols, target, out_path: Path):
    intercept = getattr(model, 'intercept_', 0.0)
    coefs = getattr(model, 'coef_', np.zeros(len(feature_cols)))
    payload = {
        'model_type': type(model).__name__,
        'intercept': float(intercept),
        'coefficients': {c: float(v) for c, v in zip(feature_cols, coefs)},
        'target': target,
        'features': feature_cols,
        'preprocessing': {'scaling': 'none', 'derived': ['receita_dia','receita_mes']},
        'inference_notes': 'Compute preco_final = preco_original * (1 - desconto_pct) before prediction.'
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main():
    init_tracking()
    df = load_dataset()
    X, y, feature_cols, target = select_features(df)
    # Validação temporal quando existir coluna de data
    if 'data' in df.columns and pd.api.types.is_datetime64_any_dtype(df['data']):
        # ordenar por tempo e sincronizar X/y
        order = np.argsort(df['data'].fillna(pd.Timestamp.min).values)
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    best_profit = -float('inf')
    best_entry = None
    # Base do experimento (ex.: context7); usaremos experimento separado por modelo
    exp_base = os.getenv("MLFLOW_EXPERIMENT_NAME", "max_receita_cafeterias")
    context_label = os.getenv("CONTEXT_LABEL") or os.getenv("CONTEXT") or "default"
    commit_sha = os.getenv("GITHUB_SHA") or os.getenv("CI_COMMIT_SHA") or "unknown"

    # Limitar a 6 modelos (exclui RandomForest para manter 6 no total)
    specs = [s for s in get_model_specs(feature_cols, target) if s.name != 'RandomForest']
    for spec in specs:
        # Definir experimento específico por modelo para registro separado no DagsHub/MLflow
        exp_name_model = f"{exp_base}_{spec.name}"
        try:
            mlflow.set_experiment(exp_name_model)
        except Exception:
            pass
        with mlflow.start_run(run_name=f"{spec.name}") as active_run:
            run_id = active_run.info.run_id
            mlflow.log_params({'model': spec.name, **spec.params})
            # Tags de contexto e commit para rastreamento no DagsHub/MLflow
            try:
                mlflow.set_tag('context', context_label)
                mlflow.set_tag('commit_sha', commit_sha)
                mlflow.set_tag('experiment', exp_name_model)
            except Exception:
                pass
            metrics = cross_validate_model(spec.estimator, X, y, cv, target)
            # Aliases compatíveis com testes: cv_rmse, cv_mae, cv_r2
            metrics_with_alias = {
                **metrics,
                'cv_rmse': metrics.get('cv_rmse_mean'),
                'cv_mae': metrics.get('cv_mae_mean'),
                'cv_r2': metrics.get('cv_r2_mean'),
            }
            mlflow.log_metrics(metrics_with_alias)

            # Fit/holdout metrics
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            spec.estimator.fit(X_train, y_train)
            y_pred = spec.estimator.predict(X_test)
            try:
                rmse_holdout = mean_squared_error(y_test, y_pred, squared=False)
            except TypeError:
                rmse_holdout = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            holdout = {
                'rmse': float(rmse_holdout),
                'r2': float(r2_score(y_test, y_pred)),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                # Métrica de negócio principal: lucro esperado médio em holdout
                'expected_profit': float(_simulate_expected_profit_holdout(spec.estimator, X_test, target)),
            }
            mlflow.log_metrics({f"holdout_{k}": v for k, v in holdout.items()})

            # Persist model locally
            local_model_path = MODELS_DIR / f"{spec.name}.pkl"
            try:
                import joblib
                joblib.dump(spec.estimator, local_model_path)
                mlflow.log_artifact(str(local_model_path))
            except Exception:
                pass

            entry = {
                'name': spec.name,
                'metrics': {**metrics_with_alias, **{f"holdout_{k}": v for k, v in holdout.items()}},
                'model_path': str(local_model_path),
                'estimator': spec.estimator,
                'run_id': run_id,
            }
            results.append(entry)

            # Seleção por lucro esperado médio em HOLDOUT (mais robusto)
            holdout_profit = holdout.get('expected_profit', -float('inf'))
            if holdout_profit > best_profit:
                best_profit = holdout_profit
                best_entry = entry

            mlflow.set_tag('selection_metric', 'holdout_expected_profit')

    assert best_entry is not None, "Nenhum modelo foi avaliado."

    # Salvar melhor modelo
    best_label = best_entry['name']
    best_model = best_entry['estimator']
    best_out = MODELS_DIR / 'best_model_max_receita.pkl'
    import joblib
    joblib.dump(best_model, best_out)
    try:
        mlflow.log_artifact(str(best_out))
    except Exception:
        pass

    # Exportar JSON para UI: se melhor for linear, exporta; caso contrário, exporta LinearRegression como baseline
    json_target_path = MODELS_DIR / 'model_linear.json'
    if isinstance(best_model, (LinearRegression, RidgeCV, LassoCV, ElasticNetCV)):
        export_linear_json(best_model, feature_cols, target, json_target_path)
    else:
        # Treinar baseline linear para inferência no front-end
        baseline = LinearRegression().fit(X, y)
        export_linear_json(baseline, feature_cols, target, json_target_path)
    try:
        mlflow.log_artifact(str(json_target_path))
    except Exception:
        pass

    # Artefato: curva média de receita/lucro vs desconto
    try:
        discounts = np.arange(0.0, 0.0400001, 0.001)
        rows = []
        for d in discounts:
            Xd = X.copy()
            if 'desconto_pct' in Xd.columns:
                Xd['desconto_pct'] = d
            if 'preco_final' in Xd.columns:
                Xd['preco_final'] = X['preco_original'] * (1.0 - d)
            qty = np.asarray(best_model.predict(Xd), dtype=float)
            price = X['preco_original'] * (1.0 - d)
            revenue = price * qty
            profit = (price - X['custo_producao']) * qty
            rows.append({
                'desconto_pct': d,
                'mean_quantity': float(np.mean(qty)),
                'mean_revenue': float(np.mean(revenue)),
                'mean_profit': float(np.mean(profit)),
            })
        curve_models_path = MODELS_DIR / 'curve_business_metric.csv'
        pd.DataFrame(rows).to_csv(curve_models_path, index=False)
        mlflow.log_artifact(str(curve_models_path))
        # Também publicar em docs/ para uso pelo site
        try:
            docs_dir = PROJECT_ROOT / 'docs'
            docs_dir.mkdir(exist_ok=True)
            curve_docs_path = docs_dir / 'curve_business_metric.csv'
            pd.DataFrame(rows).to_csv(curve_docs_path, index=False)
        except Exception as e_pub:
            warnings.warn(f"Falha ao publicar curva em docs/: {e_pub}")
    except Exception as e:
        warnings.warn(f"Falha ao gerar curva de negócio: {e}")

    # Marcar o run do melhor modelo com tags para visibilidade no DagsHub/MLflow
    try:
        from mlflow.tracking import MlflowClient
        if best_entry and best_entry.get('run_id'):
            client = MlflowClient()
            client.set_tag(best_entry['run_id'], 'is_best', 'true')
            client.set_tag(best_entry['run_id'], 'best_model', best_entry['name'])
            # Também registrar o melhor como runName claro
            client.set_tag(best_entry['run_id'], 'mlflow.runName', f"best_model_{best_entry['name']}")
            # Propagar contexto ao melhor run
            try:
                client.set_tag(best_entry['run_id'], 'context', context_label)
                client.set_tag(best_entry['run_id'], 'commit_sha', commit_sha)
            except Exception:
                pass
    except Exception as e_tag:
        warnings.warn(f"Falha ao marcar tags do best_model no MLflow: {e_tag}")

    # Métrica do melhor modelo
    best_metrics = best_entry['metrics']
    # Snapshot de métricas para testes (consumido por tests/expected_metrics.json)
    try:
        snapshot = {
            'best_model': best_label,
            'best_model_name': best_label,
            'feature_order': feature_cols,
            'target': target,
            'metrics': best_metrics,
        }
        snapshot_path = MODELS_DIR / 'metrics_snapshot.json'
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        try:
            mlflow.log_artifact(str(snapshot_path))
        except Exception:
            pass
    except Exception as e:
        warnings.warn(f"Falha ao salvar metrics_snapshot.json: {e}")
    print(f"Melhor modelo: {best_label}\nMétricas: {best_metrics}\nExport JSON: {json_target_path}")

    # SHAP: explicabilidade para modelos de árvore; para lineares já há coeficientes
    try:
        explainer = None
        model_for_shap = best_model
        if isinstance(best_model, Pipeline):
            # tenta obter o estimador final
            try:
                model_for_shap = best_model.named_steps.get('model', best_model)
            except Exception:
                model_for_shap = best_model
        if HAS_SHAP and shap is not None and (isinstance(model_for_shap, (RandomForestRegressor, GradientBoostingRegressor)) or type(model_for_shap).__name__ == 'XGBRegressor'):
            # Amostra para desempenho
            X_sample = X.sample(n=min(1000, len(X)), random_state=42)
            # Usa os dados no espaço original; alguns pipelines podem exigir transform
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_sample)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
            shap_path = MODELS_DIR / 'shap_summary.png'
            plt.tight_layout()
            plt.savefig(shap_path, dpi=140)
            plt.close()
            mlflow.log_artifact(str(shap_path))
    except Exception as e:
        warnings.warn(f"Falha ao gerar SHAP: {e}")

    # Exportar ONNX do best model e publicar em docs
    try:
        # Evita export ONNX para XGBoost, que possui suporte instável no ambiente CI
        if type(best_model).__name__ == 'XGBRegressor':
            raise RuntimeError('Export ONNX desativado para XGBRegressor')
        from skl2onnx import convert_sklearn  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
        onnx_out_models = MODELS_DIR / 'model_best.onnx'
        initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(best_model, initial_types=initial_type)
        with open(onnx_out_models, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        # Meta: ordem das features
        meta = {
            'features': feature_cols,
            'target': target,
            'model_name': best_label,
        }
        meta_models_path = MODELS_DIR / 'model_best_meta.json'
        meta_models_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        try:
            mlflow.log_artifact(str(onnx_out_models))
            mlflow.log_artifact(str(meta_models_path))
        except Exception:
            pass
        # Publicar também em docs/
        try:
            docs_dir = PROJECT_ROOT / 'docs'
            docs_dir.mkdir(exist_ok=True)
            onnx_out_docs = docs_dir / 'model_best.onnx'
            with open(onnx_out_docs, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            meta_docs_path = docs_dir / 'model_best_meta.json'
            meta_docs_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
        except Exception as e_pub:
            warnings.warn(f"Falha ao publicar ONNX/meta em docs/: {e_pub}")
    except Exception as e1:
        warnings.warn(f"Falha ao converter best model para ONNX via skl2onnx: {e1}")
        # Tentativa alternativa para XGBoost
        # Desiste para XGBoost: manter modelo em pickle e JSON linear como fallback para UI
        pass


if __name__ == '__main__':
    main()
