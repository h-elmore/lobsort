"""Extracted from spike_sort_v3.ipynb, cell 28."""

minimum_classification_probability = 0.85
classification_output_path = Path(
    f"{channel_name}_transferred_classifier_candidates.csv"
)
model_package = joblib.load(trained_model_path)
classifier = model_package["classifier"]
feature_columns = model_package["feature_columns"]
feature_medians = np.asarray(model_package["feature_medians"], dtype=float)
missing_feature_columns = [
    column for column in feature_columns if column not in feature_df.columns
]
if missing_feature_columns:
    raise KeyError(f"Missing required feature columns: {missing_feature_columns}")
X_all = feature_df[feature_columns].to_numpy(dtype=float)
invalid_rows, invalid_columns = np.where(~np.isfinite(X_all))
X_all[invalid_rows, invalid_columns] = feature_medians[invalid_columns]
predicted_labels = classifier.predict(X_all)
predicted_probabilities = classifier.predict_proba(X_all)
maximum_probabilities = np.max(predicted_probabilities, axis=1)
final_labels = np.where(
    maximum_probabilities >= minimum_classification_probability,
    predicted_labels,
    "Uncertain",
)
classified_candidates_df = feature_df[
    ["candidate_position", "sample_index", "time_seconds", "absolute_peak", "snr"]
].copy()
classified_candidates_df["predicted_label"] = predicted_labels
classified_candidates_df["classification_probability"] = maximum_probabilities
classified_candidates_df["manual_label"] = final_labels
classified_candidates_df["channel"] = channel_name
for class_index, class_name in enumerate(classifier.classes_):
    classified_candidates_df[f"probability_{class_name}"] = predicted_probabilities[
        :, class_index
    ]
classified_candidates_df.to_csv(classification_output_path, index=False)
print("\nClassifications:")
print(classified_candidates_df["manual_label"].value_counts())
print("\nSaved:", classification_output_path.resolve())
display(classified_candidates_df.head())
