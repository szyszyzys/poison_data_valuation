def load_valuation_data_from_csv(run_dir: Path) -> pd.DataFrame:
    """
    Scans seller_metrics.csv for valuation columns.
    Includes error handling for corrupted CSV lines.
    """
    csv_path = run_dir / "seller_metrics.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    try:
        # FIX: on_bad_lines='skip' tells pandas to ignore rows with formatting errors
        # instead of crashing the entire script.
        df = pd.read_csv(csv_path, on_bad_lines='skip')

        # Check if we have data
        if df.empty or 'seller_id' not in df.columns:
            return pd.DataFrame()

        # 1. Identify Seller Type
        df['type'] = df['seller_id'].apply(
            lambda x: 'Adversary' if str(x).startswith('adv') else 'Benign'
        )

        # 2. Identify Valuation Columns dynamically
        # Matches: influence_score, shapley_value, loo_score, kernelshap_score, etc.
        val_cols = [c for c in df.columns if any(x in c.lower() for x in ['influence', 'shap', 'loo'])]

        if not val_cols:
            return pd.DataFrame()

        # 3. Aggregate Average Score per Type
        summary = df.groupby('type')[val_cols].mean().reset_index()
        return summary

    except Exception as e:
        print(f"⚠️ Warning: Could not read CSV {csv_path}: {e}")
        return pd.DataFrame()