import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv


BASE_DIR = r"c:\Users\Olivia.Raykhman\OneDrive\Documents\Data Journ 1\crash data"
YEARS = [2021, 2022, 2023, 2024, 2025]

PARTIES_FILES = [
    os.path.join(BASE_DIR, f"hq1d-p-app52dopendataexport{y}parties.csv") for y in YEARS
]
CRASH_FILES = {
    y: os.path.join(BASE_DIR, f"hq1d-p-app52dopendataexport{y}crashes.csv") for y in YEARS
}
INJURED_FILE = os.path.join(BASE_DIR, "injured_witness_passengers_2021_2025_clean.csv")

OUT_DIR = os.path.join(BASE_DIR, "analysis_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

AGE_ORDER = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Unknown"]
SEX_ORDER = ["Female", "Male", "Non-binary", "Unknown"]
SEVERITY_ORDER = [
    "Fatal",
    "SuspectSerious",
    "SevereInactive",
    "SuspectMinor",
    "PossibleInjury",
    "ComplaintOfPainInactive",
    "OtherVisibleInactive",
    "Unknown",
]


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


def clean_row(row: dict) -> dict:
    return {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in row.items()}


def build_sf_collision_ids_by_year() -> tuple[dict[str, set[str]], list[int]]:
    sf_ids_by_year: dict[str, set[str]] = {}
    missing_crash_years: list[int] = []

    for year in YEARS:
        year_str = str(year)
        crash_path = CRASH_FILES[year]
        if not os.path.exists(crash_path):
            sf_ids_by_year[year_str] = set()
            missing_crash_years.append(year)
            continue

        sf_ids: set[str] = set()
        with open(crash_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                r = clean_row(row)
                if r.get("City Name", "").upper() != "SAN FRANCISCO":
                    continue
                collision_id = r.get("Collision Id", "")
                if collision_id:
                    sf_ids.add(collision_id)
        sf_ids_by_year[year_str] = sf_ids

    return sf_ids_by_year, missing_crash_years


def build_pedestrian_keys(sf_ids_by_year: dict[str, set[str]]) -> set[tuple[str, str, str]]:
    ped_keys: set[tuple[str, str, str]] = set()
    for year, path in zip(YEARS, PARTIES_FILES):
        year_str = str(year)
        sf_ids = sf_ids_by_year.get(year_str, set())
        if not sf_ids:
            continue

        cols = ["CollisionId", "PartyNumber", "PartyType"]
        part = pd.read_csv(path, usecols=cols, dtype=str, low_memory=False)
        part = part[part["PartyType"].fillna("").str.strip().eq("Pedestrian")].copy()
        part["CollisionId"] = part["CollisionId"].fillna("").str.strip()
        part["PartyNumber"] = part["PartyNumber"].fillna("").str.strip()
        part = part[part["CollisionId"].isin(sf_ids)]
        keys = set(
            zip(
                [year_str] * len(part),
                part["CollisionId"],
                part["PartyNumber"],
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
            ped.groupby(["source_year", "age_group", "sex", "severity"], dropna=False)["crash_key"]
            .nunique()
            .reset_index(name="crash_count")
        )
        frames.append(grp)

    if not frames:
        return pd.DataFrame(columns=["source_year", "age_group", "sex", "severity", "crash_count"])

    summary = pd.concat(frames, ignore_index=True)
    summary = (
        summary.groupby(["source_year", "age_group", "sex", "severity"], as_index=False)["crash_count"]
        .sum()
        .sort_values(["source_year", "age_group", "sex", "severity"])
    )

    total_by_group = summary.groupby(["source_year", "age_group", "sex"])["crash_count"].transform("sum")
    total_by_year = summary.groupby(["source_year"])["crash_count"].transform("sum")
    total_all = summary["crash_count"].sum()
    summary["percent_within_age_sex"] = (summary["crash_count"] / total_by_group * 100).round(2)
    summary["percent_of_year_ped_crashes"] = (summary["crash_count"] / total_by_year * 100).round(2)
    summary["percent_of_all_ped_crashes"] = (summary["crash_count"] / total_all * 100).round(2)
    return summary


def make_count_and_percent_plots(
    summary: pd.DataFrame,
    count_title: str,
    percent_title: str,
    count_filename: str,
    percent_filename: str,
) -> None:
    summary = summary.copy()
    if summary.empty:
        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No San Francisco pedestrian crash data available for this year.",
            ha="center",
            va="center",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, count_filename), dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(16, 7))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No San Francisco pedestrian crash data available for this year.",
            ha="center",
            va="center",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, percent_filename), dpi=180)
        plt.close(fig)
        return

    summary["age_group"] = pd.Categorical(summary["age_group"], categories=AGE_ORDER, ordered=True)
    summary["sex"] = pd.Categorical(summary["sex"], categories=SEX_ORDER, ordered=True)

    count_pivot = (
        summary.pivot_table(
            index=["age_group", "sex"],
            columns="severity",
            values="crash_count",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(columns=[c for c in SEVERITY_ORDER if c in summary["severity"].unique()], fill_value=0)
        .sort_index()
    )
    count_pivot = count_pivot.astype(float)

    fig, ax = plt.subplots(figsize=(16, 7))
    count_pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title(count_title)
    ax.set_xlabel("Age Group | Sex")
    ax.set_ylabel("Crash Count")
    ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, count_filename), dpi=180)
    plt.close(fig)

    pct_pivot = count_pivot.div(count_pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    pct_pivot = pct_pivot.fillna(0.0)
    fig, ax = plt.subplots(figsize=(16, 7))
    pct_pivot.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_title(percent_title)
    ax.set_xlabel("Age Group | Sex")
    ax.set_ylabel("Percent of Crashes")
    ax.legend(title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, percent_filename), dpi=180)
    plt.close(fig)


def main() -> None:
    sf_ids_by_year, missing_crash_years = build_sf_collision_ids_by_year()
    ped_keys = build_pedestrian_keys(sf_ids_by_year)
    summary_by_year = build_summary(ped_keys)

    overall_summary = (
        summary_by_year.groupby(["age_group", "sex", "severity"], as_index=False)["crash_count"]
        .sum()
        .sort_values(["age_group", "sex", "severity"])
    )
    total_by_group = overall_summary.groupby(["age_group", "sex"])["crash_count"].transform("sum")
    total_all = overall_summary["crash_count"].sum()
    overall_summary["percent_within_age_sex"] = (overall_summary["crash_count"] / total_by_group * 100).round(2)
    overall_summary["percent_of_all_ped_crashes"] = (overall_summary["crash_count"] / total_all * 100).round(2)

    summary_csv = os.path.join(OUT_DIR, "pedestrian_crash_summary_table.csv")
    overall_summary.to_csv(summary_csv, index=False)
    summary_by_year_csv = os.path.join(OUT_DIR, "pedestrian_crash_summary_table_by_year.csv")
    summary_by_year.to_csv(summary_by_year_csv, index=False)

    make_count_and_percent_plots(
        overall_summary,
        "Pedestrian Crashes by Age Group, Sex, and Severity (Count)",
        "Pedestrian Crash Severity Mix by Age Group and Sex (Percent)",
        "pedestrian_crashes_count_by_age_sex_severity.png",
        "pedestrian_crashes_percent_by_age_sex_severity.png",
    )

    for year in YEARS:
        year_str = str(year)
        year_summary = summary_by_year[summary_by_year["source_year"] == year_str]
        make_count_and_percent_plots(
            year_summary,
            f"Pedestrian Crashes by Age Group, Sex, and Severity ({year_str}, Count)",
            f"Pedestrian Crash Severity Mix by Age Group and Sex ({year_str}, Percent)",
            f"pedestrian_crashes_{year_str}_count_by_age_sex_severity.png",
            f"pedestrian_crashes_{year_str}_percent_by_age_sex_severity.png",
        )

    sf_collision_total = sum(len(v) for v in sf_ids_by_year.values())
    print(f"San Francisco collisions (from crash files): {sf_collision_total:,}")
    print(f"San Francisco pedestrian party keys: {len(ped_keys):,}")
    print(f"Overall summary rows: {len(overall_summary):,}")
    print(f"Year-level summary rows: {len(summary_by_year):,}")
    print(f"Summary table: {summary_csv}")
    print(f"Year-level summary table: {summary_by_year_csv}")
    print(
        "Overall plots: "
        + os.path.join(OUT_DIR, "pedestrian_crashes_count_by_age_sex_severity.png")
        + " ; "
        + os.path.join(OUT_DIR, "pedestrian_crashes_percent_by_age_sex_severity.png")
    )
    print("Yearly plots created for: " + ", ".join(str(y) for y in YEARS))
    if missing_crash_years:
        print(
            "Missing crash files (excluded from SF filtering): "
            + ", ".join(str(y) for y in missing_crash_years)
        )


if __name__ == "__main__":
    main()
