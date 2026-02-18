from pathlib import Path

import pandas as pd


YEARS = [2021, 2022, 2023, 2024, 2025]


def norm(column_name: str) -> str:
    return "".join(ch for ch in str(column_name).strip().lower() if ch.isalnum())


def build_address(row: pd.Series) -> str:
    primary = str(row.get("PrimaryRoad", "")).strip()
    secondary = str(row.get("SecondaryRoad", "")).strip()
    sec_direction = str(row.get("SecondaryDirection", "")).strip()
    sec_distance = str(row.get("SecondaryDistance", "")).strip()

    if primary and secondary and secondary.lower() != "nan":
        return f"{primary} & {secondary}"
    if primary:
        if (
            sec_distance
            and sec_distance.lower() != "nan"
            and sec_direction
            and sec_direction.lower() != "nan"
        ):
            return f"{sec_direction} {sec_distance} from {primary}"
        return primary
    return ""


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "analysis_outputs"
    output_dir.mkdir(exist_ok=True)

    all_non_driver_ages = []
    yearly_bicyclist_counts = []
    bike_location_frames = []

    for year in YEARS:
        crashes_path = base_dir / f"hq1d-p-app52dopendataexport{year}crashes.csv"
        parties_path = base_dir / f"hq1d-p-app52dopendataexport{year}parties.csv"

        crashes = pd.read_csv(crashes_path, low_memory=False)
        crash_cols = {norm(c): c for c in crashes.columns}

        col_collision = crash_cols.get("collisionid")
        col_city = crash_cols.get("cityname")
        col_primary = crash_cols.get("primaryroad")
        col_secondary = crash_cols.get("secondaryroad")
        col_secdir = crash_cols.get("secondarydirection")
        col_secdist = crash_cols.get("secondarydistance")
        col_lat = crash_cols.get("latitude")
        col_lon = crash_cols.get("longitude")
        col_dt = crash_cols.get("crashdatetime")

        if col_collision is None or col_city is None:
            raise ValueError(f"Required crash columns missing in {crashes_path.name}")

        sf_crashes = crashes[
            crashes[col_city].astype(str).str.strip().str.upper().eq("SAN FRANCISCO")
        ].copy()
        sf_crashes[col_collision] = sf_crashes[col_collision].astype(str).str.strip()
        sf_collision_ids = set(sf_crashes[col_collision].dropna().tolist())

        chunk_iter = pd.read_csv(
            parties_path,
            chunksize=200_000,
            engine="python",
            on_bad_lines="skip",
        )

        non_driver_age_rows = []
        bicyclist_party_count = 0
        bike_collision_ids = set()
        sf_party_row_count = 0

        for chunk in chunk_iter:
            party_cols = {norm(c): c for c in chunk.columns}
            p_col_collision = party_cols.get("collisionid")
            p_col_partytype = party_cols.get("partytype")
            p_col_age = party_cols.get("statedage")

            if p_col_collision is None or p_col_partytype is None:
                raise ValueError(f"Required party columns missing in {parties_path.name}")

            chunk[p_col_collision] = chunk[p_col_collision].astype(str).str.strip()
            sf_parties = chunk[chunk[p_col_collision].isin(sf_collision_ids)].copy()
            if sf_parties.empty:
                continue

            sf_party_row_count += len(sf_parties)
            party_type_upper = (
                sf_parties[p_col_partytype].astype(str).str.strip().str.upper()
            )

            non_driver_mask = ~party_type_upper.eq("DRIVER")
            if p_col_age is not None:
                ages = sf_parties.loc[non_driver_mask, [p_col_age]].copy()
                ages.columns = ["StatedAge"]
                non_driver_age_rows.append(ages)

            bike_mask = party_type_upper.str.contains("BICYCL", na=False)
            bicyclist_party_count += int(bike_mask.sum())

            if bike_mask.any():
                bike_collision_ids.update(
                    sf_parties.loc[bike_mask, p_col_collision]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )

        if non_driver_age_rows:
            all_non_driver_ages.append(pd.concat(non_driver_age_rows, ignore_index=True))

        yearly_bicyclist_counts.append(
            {
                "Year": year,
                "BicyclistPartyCount": bicyclist_party_count,
                "UniqueBikeCrashCount": len(bike_collision_ids),
                "SFCollisionCount": len(sf_collision_ids),
                "SFPartyRowCount": sf_party_row_count,
            }
        )

        if bike_collision_ids:
            location_cols = [
                c
                for c in [
                    col_collision,
                    col_primary,
                    col_secondary,
                    col_secdir,
                    col_secdist,
                    col_lat,
                    col_lon,
                    col_dt,
                ]
                if c is not None
            ]
            bike_locations = sf_crashes[
                sf_crashes[col_collision].astype(str).str.strip().isin(bike_collision_ids)
            ][location_cols].copy()
            bike_locations["Year"] = year

            rename_map = {
                col_collision: "CollisionId",
                col_primary: "PrimaryRoad",
                col_secondary: "SecondaryRoad",
                col_secdir: "SecondaryDirection",
                col_secdist: "SecondaryDistance",
                col_lat: "Latitude",
                col_lon: "Longitude",
                col_dt: "CrashDateTime",
            }
            bike_locations = bike_locations.rename(
                columns={k: v for k, v in rename_map.items() if k in bike_locations.columns}
            )

            bike_locations["Address"] = bike_locations.apply(build_address, axis=1)
            keep_cols = [
                c
                for c in [
                    "Year",
                    "CollisionId",
                    "Address",
                    "PrimaryRoad",
                    "SecondaryRoad",
                    "Latitude",
                    "Longitude",
                    "CrashDateTime",
                ]
                if c in bike_locations.columns
            ]
            bike_location_frames.append(
                bike_locations[keep_cols].drop_duplicates(subset=["CollisionId"])
            )

    if all_non_driver_ages:
        ages = pd.concat(all_non_driver_ages, ignore_index=True)
        ages["StatedAge"] = ages["StatedAge"].astype(str).str.strip()
        ages.loc[
            ages["StatedAge"].isin(["", "nan", "None"]), "StatedAge"
        ] = "(blank)"

        age_frequency = (
            ages.groupby("StatedAge", dropna=False)
            .size()
            .reset_index(name="Frequency")
        )

        # Normalize whole-number floats (for example "25.0" -> "25"), then regroup.
        age_frequency["StatedAge"] = age_frequency["StatedAge"].apply(
            lambda v: str(v)[:-2]
            if str(v).endswith(".0") and str(v)[:-2].replace("-", "").isdigit()
            else str(v)
        )
        age_frequency = (
            age_frequency.groupby("StatedAge", as_index=False)["Frequency"]
            .sum()
            .reset_index(drop=True)
        )

        age_frequency["_age_num"] = pd.to_numeric(
            age_frequency["StatedAge"], errors="coerce"
        )
        age_frequency = (
            age_frequency.sort_values(
                ["_age_num", "StatedAge"], na_position="last"
            )
            .drop(columns=["_age_num"])
            .reset_index(drop=True)
        )

        nonblank_age_frequency = age_frequency[
            ~age_frequency["StatedAge"].isin(["(blank)", "", "nan", "None"])
        ].copy()
    else:
        age_frequency = pd.DataFrame(columns=["StatedAge", "Frequency"])
        nonblank_age_frequency = age_frequency.copy()

    bicyclist_counts = pd.DataFrame(yearly_bicyclist_counts).sort_values(
        "Year"
    ).reset_index(drop=True)

    if bike_location_frames:
        bike_locations_final = pd.concat(bike_location_frames, ignore_index=True)
        bike_locations_final = bike_locations_final.sort_values(
            ["Year", "CollisionId"]
        ).reset_index(drop=True)
    else:
        bike_locations_final = pd.DataFrame(
            columns=[
                "Year",
                "CollisionId",
                "Address",
                "PrimaryRoad",
                "SecondaryRoad",
                "Latitude",
                "Longitude",
                "CrashDateTime",
            ]
        )

    age_path = output_dir / "sf_non_driver_stated_age_frequency_2021_2025.csv"
    age_nonblank_path = (
        output_dir / "sf_non_driver_stated_age_frequency_nonblank_2021_2025.csv"
    )
    counts_path = output_dir / "sf_bicyclist_involved_counts_by_year_2021_2025.csv"
    bike_locations_path = output_dir / "sf_bicycle_crash_locations_2021_2025.csv"

    age_frequency.to_csv(age_path, index=False)
    nonblank_age_frequency.to_csv(age_nonblank_path, index=False)
    bicyclist_counts.to_csv(counts_path, index=False)
    bike_locations_final.to_csv(bike_locations_path, index=False)

    print("Wrote output files:")
    print(age_path)
    print(age_nonblank_path)
    print(counts_path)
    print(bike_locations_path)
    print()
    print("Yearly bicyclist summary:")
    print(bicyclist_counts.to_string(index=False))


if __name__ == "__main__":
    main()
