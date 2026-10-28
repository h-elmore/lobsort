"""Extracted from spike_sort_v3.ipynb, cell 37."""

required_label_columns = {"sample_index", "manual_label"}
missing_columns = required_label_columns - set(manual_labels_df.columns)
if missing_columns:
    raise KeyError(f"manual_labels_df is missing: {missing_columns}")
training_labels_df = manual_labels_df[["sample_index", "manual_label"]].copy()
training_labels_df["sample_index"] = pd.to_numeric(
    training_labels_df["sample_index"], errors="coerce"
)
training_labels_df = training_labels_df.dropna(
    subset=["sample_index", "manual_label"]
).copy()
training_labels_df["sample_index"] = training_labels_df["sample_index"].astype(int)
training_labels_df = training_labels_df.drop_duplicates(
    subset="sample_index", keep="last"
)
labeled_features_df = candidate_features_df.merge(
    training_labels_df, on="sample_index", how="inner"
)
labeled_features_df = labeled_features_df[labeled_features_df["valid_waveform"]].copy()
if len(labeled_features_df) == 0:
    raise ValueError(
        "No manually labeled candidates matched the candidate feature table."
    )
label_counts = labeled_features_df["manual_label"].value_counts()
print("Training examples by class:")
display(label_counts.rename("number_of_examples"))
if len(label_counts) < 2:
    raise ValueError("At least two manual-label classes are required.")
maximum_possible_pcs = min(
    number_of_waveform_pcs,
    len(waveform_feature_columns),
    max(1, len(labeled_features_df) - 1),
)
waveform_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler()),
        (
            "pca",
            PCA(n_components=maximum_possible_pcs, random_state=classifier_random_seed),
        ),
    ]
)
context_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]
)
feature_preprocessor = ColumnTransformer(
    [
        ("waveform_shape", waveform_pipeline, waveform_feature_columns),
        ("temporal_and_rhythm", context_pipeline, context_feature_columns),
    ]
)
classifier = RandomForestClassifier(
    n_estimators=number_of_trees,
    min_samples_leaf=minimum_samples_per_leaf,
    class_weight="balanced",
    random_state=classifier_random_seed,
    n_jobs=-1,
    max_features="sqrt",
)
context_classifier = Pipeline(
    [("features", feature_preprocessor), ("classifier", classifier)]
)
X_labeled = labeled_features_df[waveform_feature_columns + context_feature_columns]
y_labeled = labeled_features_df["manual_label"].astype(str)
number_of_groups = min(number_of_validation_time_blocks, len(labeled_features_df))
if number_of_groups < 2:
    raise ValueError("Not enough labeled examples for validation.")
time_group_edges = np.linspace(
    labeled_features_df["time_seconds"].min(),
    labeled_features_df["time_seconds"].max(),
    number_of_groups + 1,
)
validation_groups = np.clip(
    np.digitize(labeled_features_df["time_seconds"], time_group_edges[1:-1]),
    0,
    number_of_groups - 1,
)
unique_validation_groups = np.unique(validation_groups)
if len(unique_validation_groups) >= 2:
    group_cv = GroupKFold(n_splits=len(unique_validation_groups))
    smallest_training_fold = min(
        int(np.sum(validation_groups != g)) for g in unique_validation_groups
    )
    context_classifier.set_params(
        features__waveform_shape__pca__n_components=min(
            maximum_possible_pcs, max(1, smallest_training_fold)
        )
    )
    cross_validated_predictions = cross_val_predict(
        context_classifier,
        X_labeled,
        y_labeled,
        groups=validation_groups,
        cv=group_cv,
        method="predict",
        n_jobs=1,
    )
    print("\nTime-blocked validation results:\n")
    print(
        classification_report(y_labeled, cross_validated_predictions, zero_division=0)
    )
    validation_balanced_accuracy = balanced_accuracy_score(
        y_labeled, cross_validated_predictions
    )
    print("Balanced validation accuracy:", round(validation_balanced_accuracy, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_labeled, cross_validated_predictions, cmap="Blues", xticks_rotation=45
    )
    plt.title("Time-blocked validation confusion matrix")
    plt.tight_layout()
    plt.show()
else:
    print(
        "Could not perform time-blocked validation. The labeled examples occupy only one time block."
    )
context_classifier.fit(X_labeled, y_labeled)
print(
    f"\nContext-aware classifier fitted using {len(labeled_features_df)} manually labeled examples."
)
