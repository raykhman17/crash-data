import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = r"c:\Users\Olivia.Raykhman\OneDrive\Documents\Data Journ 1\crash data"
YEARS = [2021, 2022, 2023, 2024, 2025]

PARTIES_FILES = [
    os.path.join(BASE_DIR, f"hq1d-p-app52dopendataexport{y}parties.csv") for y in YEARS
]
INJURED_FILE = os.path.join(BASE_DIR, "injured_witness_passengers_2021_2025_clean.csv")

OUT_DIR = os.path.join(BASE_DIR, "analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def to_age_group(age_value: str) -> str:
    try:
        age = float(str(age_value).strip())
    except Exception:
        return "Unknown"

    if age < 0:
        return "Unknown"
    if age < 18:
        return "0-17"
    if age <= 24:
        return "18-24"
    if age <= 34:
        return "25-34"
    if age <= 44:
        return "35-44"
    if age <= 54:
        return "45-54"
    if age <= 64:
        return "55-64"
    return "65+"


def normalize_sex(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip().upper()
    if v in {"M", "MALE"}:
        return "Male"
    if v in {"F", "FEMALE"}:
        return "Female"
    if v in {"NON-BINARY", "NONBINARY", "X"}:
        return "Non-binary"
    return "Unknown"


def normalize_severity(value: str) -> str:
    if pd.isna(value):
        return "Unknown"
    v = str(value).strip()
    if not v:
        return "Unknown"
    return v


def build_pedestrian_keys() -> set[tuple[str, str, str]]:
    ped_keys: set[tuple[str, str, str]] = set()
    for year, path in zip(YEARS, PARTIES_FILES):
        cols = ["CollisionId", "PartyNumber", "PartyType"]
        part = pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
        part = part[part["PartyType"].fillna("").str.strip().eq("Pedestrian")]
        keys = set(
            zip(
                [str(year)] * len(part),
                part["CollisionId"].fillna("").str.strip(),
                part["PartyNumber"].fillna("").str.strip(),
            )
        )
        ped_keys.update(keys)
    return ped_keys


def build_summary(ped_keys: set[tuple[str, str, str]]) -> pd.DataFrame:
    usecols = [
        "collision_id",
        "party_number",
        "stated_age",
        "gender_desc",
        "extent_of_injury_code",
        "source_year",
    ]

    frames = []
    for chunk in pd.read_csv(
        INJURED_FILE,
        usecols=usecols,
        dtype=str,
        chunksize=300000,
        low_memory=False,
    ):
        chunk = chunk.copy()
        chunk["source_year"] = chunk["source_year"].fillna("").str.strip()
        chunk["collision_id"] = chunk["collision_id"].fillna("").str.strip()
        chunk["party_number"] = chunk["party_number"].fillna("").str.strip()

        key_series = list(zip(chunk["source_year"], chunk["collision_id"], chunk["party_number"]))
        mask = pd.Series(key_series).isin(ped_keys).to_numpy()
        ped = chunk.loc[mask].copy()
        if ped.empty:
            continue

        ped["age_group"] = ped["stated_age"].map(to_age_group)
        ped["sex"] = ped["gender_desc"].map(normalize_sex)
        ped["severity"] = ped["extent_of_injury_code"].map(normalize_severity)
        ped["crash_key"] = ped["source_year"] + "-" + ped["collision_id"]

        # Count unique crashes per group/severity.
        grp = (
            ped.groupby(["age_group", "sex", "severity"], dropna=False)["crash_key"]
            .nunique()
            .reset_index(name="crash_count")
        )
        frames.append(grp)

    if not frames:
        return pd.DataFrame(columns=["age_group", "sex", "severity", "crash_count"])

    summary = pd.concat(frames, ignore_index=True)
    summary = (
        summary.groupby(["age_group", "sex", "severity"], as_index=False)["crash_count"]
        .sum()
        .sort_values(["age_group", "sex", "severity"])
    )

    total_by_group = summary.groupby(["age_group", "sex"])["crash_count"].transform("sum")
    total_all = summary["crash_count"].sum()
    summary["percent_within_age_sex"] = (summary["crash_count"] / total_by_group * 100).round(2)
    summary["percent_of_all_ped_crashes"] = (summary["crash_count"] / total_all * 100).round(2)
    return summary


def make_plots(summary: pd.DataFrame) -> None:
    age_order = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Unknown"]
    sex_order = ["Female", "Male", "Non-binary", "Unknown"]
    severity_order = [
        "Fatal",
        "SuspectSerious",
        "SevereInactive",
        "SuspectMinor",
        "PossibleInjury",
        "ComplaintOfPainInactive",
        "OtherVisibleInactive",
        "Unknown",
    ]

    summary = summary.copy()
    if summary.empty:
        return

    summary["age_group"] = pd.Categorical(summary["age_group"], categories=age_order, ordered=True)
    summary["sex"] = pd.Categorical(summary["sex"], categories=sex_order, ordered=True)

    # Count plot.
    count_pivot = (
        summary.pivot_table(
            index=["age_group", "sex"],
            columns="severity",
            values="crash_count",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=[c for c in severity_order if c in summary["severity"].unique()], fill_value=0)
        .sort_index()
    )
    count_pivot = count_pivot.astype(float)

    fig, ax = plt.subplots(figsize=(16, 7))
    count_pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title("Pedestrian Crashes by Age Group, Sex, and Severity (Count)")
    ax.set_xlabel("Age Group | Sex")
    ax.set_ylabel("Crash Count")
    ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pedestrian_crashes_count_by_age_sex_severity.png"), dpi=180)
    plt.close(fig)

    # Percent plot.
    pct_pivot = count_pivot.div(count_pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pct_pivot = pct_pivot.fillna(0.0)
    fig, ax = plt.subplots(figsize=(16, 7))
    pct_pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title("Pedestrian Crash Severity Mix by Age Group and Sex (Percent)")
    ax.set_xlabel("Age Group | Sex")
    ax.set_ylabel("Percent of Crashes")
    ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pedestrian_crashes_percent_by_age_sex_severity.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    ped_keys = build_pedestrian_keys()
    summary = build_summary(ped_keys)

    summary_csv = os.path.join(OUT_DIR, "pedestrian_crash_summary_table.csv")
    summary.to_csv(summary_csv, index=False)

    make_plots(summary)

    print(f"Pedestrian party keys: {len(ped_keys):,}")
    print(f"Summary rows: {len(summary):,}")
    print(f"Summary table: {summary_csv}")
    print(
        "Plots: "
        + os.path.join(OUT_DIR, "pedestrian_crashes_count_by_age_sex_severity.png")
        + " ; "
        + os.path.join(OUT_DIR, "pedestrian_crashes_percent_by_age_sex_severity.png")
    )


if __name__ == "__main__":
    main()
