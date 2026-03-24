"""
Phase 4: Model Evaluation and Comparison
Evaluate trained models, generate visualizations, and compare performance.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from utils_io import save_dataframe
import os

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str, config) -> dict:
    # Evaluate trained model and calculate performance metrics
    logging.info(f"Evaluating {model_name}...")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    # Log metrics
    logging.info(f"--- {model_name} Performance ---")
    for metric, value in metrics.items():
        if metric != 'model':
            logging.info(f"  {metric}: {value:.4f}")

    return metrics

def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray,
                         model_name: str, config) -> None:
    # Create and save confusion matrix heatmap
    logging.info(f"Creating confusion matrix for {model_name}...")

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    plt.figure(figsize=config.PLOT_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Away Win', 'Home Win'],
                yticklabels=['Away Win', 'Home Win'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save figure
    filepath = os.path.join(config.PLOTS_DIR, f'{model_name}_confusion_matrix.png')
    plt.savefig(filepath, dpi=config.PLOT_DPI)
    plt.close()

    logging.info(f"Saved confusion matrix to {filepath}")

def plot_feature_importance(model, feature_names: list, model_name: str, config) -> None:
    # Create and save feature importance plot
    logging.info(f"Creating feature importance plot for {model_name}...")

    # Get feature importances
    importances = model.feature_importances_

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select top N features
    top_features = importance_df.head(config.TOP_N_FEATURES)

    # Create figure
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {config.TOP_N_FEATURES} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save figure
    filepath = os.path.join(config.PLOTS_DIR, f'{model_name}_feature_importance.png')
    plt.savefig(filepath, dpi=config.PLOT_DPI)
    plt.close()

    logging.info(f"Saved feature importance plot to {filepath}")

    # Also log top 10 features
    logging.info(f"Top 10 features for {model_name}:")
    for idx, row in importance_df.head(10).iterrows():
        logging.info(f"  {row['feature']}: {row['importance']:.4f}")


def plot_roc_curve(y_true: pd.Series, y_proba: np.ndarray,
                   model_name: str, config) -> None:
    # Create and save ROC curve plot
    logging.info(f"Creating ROC curve for {model_name}...")

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Create figure
    plt.figure(figsize=config.PLOT_FIGSIZE)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save figure
    filepath = os.path.join(config.PLOTS_DIR, f'{model_name}_roc_curve.png')
    plt.savefig(filepath, dpi=config.PLOT_DPI)
    plt.close()

    logging.info(f"Saved ROC curve to {filepath}")


def plot_hyperparameter_tuning(cv_results: dict, param_name: str,
                               model_name: str, config) -> None:
    # Create and save hyperparameter tuning plot
    try:
        logging.info(f"Creating hyperparameter tuning plot for {model_name} - {param_name}...")

        # Extract results
        results_df = pd.DataFrame(cv_results)

        # Filter to rows where only the target parameter varies
        param_key = f'param_{param_name}'
        if param_key not in results_df.columns:
            logging.warning(f"Parameter {param_name} not found in CV results")
            return

        # Group by parameter value and get mean scores
        param_scores = results_df.groupby(param_key)['mean_test_score'].mean().reset_index()
        param_scores = param_scores.sort_values(param_key)

        # Create figure
        plt.figure(figsize=config.PLOT_FIGSIZE)
        plt.plot(param_scores[param_key].astype(str), param_scores['mean_test_score'],
                 marker='o', linewidth=2, markersize=8)
        plt.xlabel(param_name)
        plt.ylabel('Mean CV Accuracy')
        plt.title(f'Hyperparameter Tuning: {param_name} - {model_name}')
        plt.xticks(rotation=45)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Save figure
        filepath = os.path.join(config.PLOTS_DIR, f'{model_name}_{param_name}_tuning.png')
        plt.savefig(filepath, dpi=config.PLOT_DPI)
        plt.close()

        logging.info(f"Saved hyperparameter tuning plot to {filepath}")

    except Exception as e:
        logging.warning(f"Could not create hyperparameter plot for {param_name}: {str(e)}")


def compare_models(rf_metrics: dict, xgb_metrics: dict, config) -> pd.DataFrame:
    # Compare performance of Random Forest and XGBoost models.
    logging.info("Comparing model performance...")

    # Create comparison DataFrame
    comparison = pd.DataFrame([rf_metrics, xgb_metrics])
    comparison = comparison.set_index('model')

    # Add winner column for each metric
    for col in comparison.columns:
        max_val = comparison[col].max()
        comparison[f'{col}_winner'] = comparison[col].apply(
            lambda x: '★' if x == max_val else ''
        )

    # Save comparison
    save_dataframe(comparison, config.MODEL_COMPARISON_PATH, index=True)

    # Log formatted comparison
    logging.info("\nMODEL COMPARISON")
    logging.info(f"\n{comparison.to_string()}")

    return comparison


def evaluate_all_models(models_dict: dict, X_test: pd.DataFrame,
                        y_test: pd.Series, feature_names: list, config) -> pd.DataFrame:
    # Evaluate all trained models and generate comparison.
    logging.info("\nEVALUATING ALL MODELS")

    all_metrics = []

    # Evaluate Random Forest
    rf_model = models_dict['random_forest']['model']
    rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest', config)
    all_metrics.append(rf_metrics)

    # Generate plots for Random Forest
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred_rf, 'random_forest', config)
    plot_feature_importance(rf_model, feature_names, 'random_forest', config)
    plot_roc_curve(y_test, y_proba_rf, 'random_forest', config)

    # Plot hyperparameter tuning for key parameters
    rf_cv_results = models_dict['random_forest']['cv_results']
    for param in ['n_estimators', 'max_depth']:
        plot_hyperparameter_tuning(rf_cv_results, param, 'random_forest', config)

    # Evaluate XGBoost
    xgb_model = models_dict['xgboost']['model']
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, 'XGBoost', config)
    all_metrics.append(xgb_metrics)

    # Generate plots for XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred_xgb, 'xgboost', config)
    plot_feature_importance(xgb_model, feature_names, 'xgboost', config)
    plot_roc_curve(y_test, y_proba_xgb, 'xgboost', config)

    # Plot hyperparameter tuning for key parameters
    xgb_cv_results = models_dict['xgboost']['cv_results']
    for param in ['n_estimators', 'max_depth', 'learning_rate']:
        plot_hyperparameter_tuning(xgb_cv_results, param, 'xgboost', config)

    # Compare models
    comparison = compare_models(rf_metrics, xgb_metrics, config)

    logging.info("\nEVALUATION COMPLETE")

    return comparison