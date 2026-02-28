import pandas as pd


def load_dataset(input_path, required_cols):
    df = pd.read_csv(input_path, low_memory=False, usecols=required_cols)
    return df


def detect_subtype_column(columns):
    candidates = ["attack_subcat", "attack_subtype", "subtype", "subcat"]
    for col in candidates:
        if col in columns:
            return col
    return None


def main():
    input_path = "datasets/UNSW-NB15_cleaned.csv"
    label_col = "label"
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    subtype_col = detect_subtype_column(header)
    usecols = [label_col, "attack_cat"]
    if subtype_col:
        usecols.append(subtype_col)
    df = load_dataset(input_path, usecols)

    y = df[label_col]
    safe_count = int((y == 0).sum())
    anomaly_count = int((y == 1).sum())
    total = int(len(y))

    print(f"Total samples: {total}")
    print(f"Safe (label=0): {safe_count}")
    print(f"Anomalies (label=1): {anomaly_count}")
    if safe_count == 0:
        print("Anomaly/Clean ratio: inf (no clean samples)")
    else:
        ratio = anomaly_count / safe_count
        print(f"Anomaly/Clean ratio: {ratio:.6f}")

    attack_series = df["attack_cat"].dropna().astype(str).str.strip()
    attack_series = attack_series[attack_series != ""]
    attack_types = sorted(attack_series.unique().tolist())
    print(f"Attack types (unique): {len(attack_types)}")

    counts = attack_series.value_counts()
    for attack_type in counts.index.tolist():
        print(f"{attack_type}: {int(counts[attack_type])}")

    if subtype_col:
        print(f"Subtype column: {subtype_col}")
        subtype_series = df[subtype_col].dropna().astype(str).str.strip()
        subtype_series = subtype_series[subtype_series != ""]
        sub_counts = df.assign(_sub=subtype_series).dropna(subset=["attack_cat", "_sub"])
        if len(sub_counts) == 0:
            print("No subtype values found.")
        else:
            grouped = sub_counts.groupby("attack_cat")["_sub"].nunique().sort_index()
            for attack_type in grouped.index.tolist():
                print(f"{attack_type} subtypes: {int(grouped[attack_type])}")
    else:
        print("No subtype column found in the dataset.")


if __name__ == "__main__":
    main()
