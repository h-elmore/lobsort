"""Extracted from spike_sort_v3.ipynb, cell 44."""

prediction_source_df = classified_candidates_df.copy()
print("Available transferred columns:")
print(prediction_source_df.columns.tolist())
required_source_columns = {
    "sample_index",
    "predicted_label",
    "classification_probability",
}
missing_source_columns = required_source_columns - set(prediction_source_df.columns)
if missing_source_columns:
    raise KeyError(
        f"classified_candidates_df is missing columns: {missing_source_columns}"
    )
prediction_source_df["sample_index"] = pd.to_numeric(
    prediction_source_df["sample_index"], errors="coerce"
)
prediction_source_df["classification_probability"] = pd.to_numeric(
    prediction_source_df["classification_probability"], errors="coerce"
)
prediction_source_df = prediction_source_df.dropna(subset=["sample_index"]).copy()
prediction_source_df["sample_index"] = prediction_source_df["sample_index"].astype(int)
prediction_source_df = prediction_source_df.drop_duplicates(
    subset="sample_index", keep="last"
)
probability_columns = [
    column
    for column in prediction_source_df.columns
    if column.startswith("probability_")
]
transfer_columns = [
    "sample_index",
    "predicted_label",
    "classification_probability",
    *probability_columns,
]
columns_to_remove_before_merge = [
    "predicted_label",
    "predicted_label_x",
    "predicted_label_y",
    "classification_probability",
    "classification_probability_x",
    "classification_probability_y",
    "prediction_probability",
]
columns_to_remove_before_merge.extend(
    [
        column
        for column in candidate_features_df.columns
        if column.startswith("probability_")
    ]
)
columns_to_remove_before_merge = [
    column
    for column in columns_to_remove_before_merge
    if column in candidate_features_df.columns
]
if columns_to_remove_before_merge:
    candidate_features_df = candidate_features_df.drop(
        columns=columns_to_remove_before_merge
    )
candidate_features_df["sample_index"] = pd.to_numeric(
    candidate_features_df["sample_index"], errors="coerce"
)
candidate_features_df = candidate_features_df.dropna(subset=["sample_index"]).copy()
candidate_features_df["sample_index"] = candidate_features_df["sample_index"].astype(
    int
)
candidate_features_df = candidate_features_df.merge(
    prediction_source_df[transfer_columns],
    on="sample_index",
    how="left",
    validate="one_to_one",
)
matched_predictions = candidate_features_df["predicted_label"].notna()
print("Rhythmic feature candidates:", len(candidate_features_df))
print("Transferred candidate rows:", len(prediction_source_df))
print("Matched predictions:", int(matched_predictions.sum()))
print("Unmatched candidates:", int((~matched_predictions).sum()))
print("\nMatched predicted labels:")
print(candidate_features_df["predicted_label"].value_counts(dropna=False))
