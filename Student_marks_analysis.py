
"""Student Exam Marks Analysis.

This module analyzes student examination performance using Pandas and NumPy.
It supports:
1. Interactive user input for student names and subject marks.
2. Loading marks from a CSV dataset.
3. Automated data cleaning and imputation for missing marks.
4. Comprehensive statistical analysis (Mean, Median, Std Dev, Percentiles).
5. Grade assignment (A+, A, B, C, D, F) and Pass/Fail status.
6. Exporting analyzed results to CSV files.

Compatible with terminal execution, Google Colab, and Jupyter Notebooks.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

# List of subjects analyzed
SUBJECTS: List[str] = [  "Math",
    "Physics",
    "Chemistry",
    "Computer_Science",
    "English",
]
MAX_MARKS_PER_SUBJECT: int = 100
PASS_MARK_PER_SUBJECT: int = 40
PASS_PERCENTAGE: float = 40.0

# Embedded sample dataset for fallback / testing
SAMPLE_CSV_CONTENT = """Student_ID,Name,Math,Physics,Chemistry,Computer_Science,English
STU001,Aarav Sharma,88,92,85,95,78
STU002,Diya Patel,95,89,94,98,91
STU003,Rohan Verma,72,68,75,80,70
STU004,Ananya Iyer,91,94,,92,89
STU005,Kabir Mehta,45,52,48,50,55
STU006,Ishita Gupta,84,79,88,85,82
STU007,Vivaan Reddy,35,42,38,40,45
STU008,Sanya Malhotra,78,82,80,,76
STU009,Arjun Nair,96,98,95,99,94
STU010,Meera Joshi,60,65,58,62,64
STU011,Aditya Rao,28,34,30,42,38
STU012,Pooja Sen,82,85,89,90,86
STU013,Karan Bhatia,74,71,69,75,72
STU014,Tanvi Deshmukh,89,91,90,94,88
STU015,Rishi Kulkarni,54,58,52,60,56
STU016,Neha Kapoor,92,88,91,95,90
STU017,Siddharth Das,67,70,72,68,65
STU018,Priyanka Roy,40,45,42,48,50
"""


def resolve_paths(base_path: Path | None = None) -> Tuple[Path, Path, Path]:
    """Resolve project directory paths for data and output files.

    Works seamlessly in standard scripts, Jupyter notebooks, Google Colab,
    and interactive REPL sessions.
    """
    if base_path is not None:
        base_dir = Path(base_path).resolve()
    else:
        try:
            base_dir = Path(__file__).resolve().parent
        except NameError:
            # Fallback for Jupyter / Google Colab / interactive sessions
            cwd = Path.cwd()
            if cwd == Path("/"):
                known_path = Path("/Users/adarshpandey/.gemini/antigravity/scratch/marks-analysis")
                base_dir = known_path if known_path.exists() else Path.home() / "marks-analysis"
            else:
                base_dir = cwd

    data_dir = base_dir / "data"
    output_dir = base_dir / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = data_dir / "sample_marks.csv"
    analyzed_file = output_dir / "analyzed_marks.csv"
    summary_file = output_dir / "subject_summary.csv"

    # Automatically generate sample data if missing
    if not data_file.exists():
        data_file.write_text(SAMPLE_CSV_CONTENT.strip(), encoding="utf-8")
        print(f"[Setup] Created sample dataset at: {data_file}")

    return data_file, analyzed_file, summary_file


def get_valid_mark(subject: str) -> float | None:
    """Prompt the user for a subject mark between 0 and 100, or None if blank."""
    while True:
        try:
            raw_val = input(f"    - {subject} (0-100, or press Enter for missing): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw_val == "" or raw_val.upper() in ("NA", "NONE", "NULL", "NAN", "-"):
            return None

        try:
            val = float(raw_val)
            if 0.0 <= val <= 100.0:
                return val
            print("      [!] Score must be between 0 and 100. Please try again.")
        except ValueError:
            print("      [!] Invalid number. Enter a score between 0 and 100, or press Enter.")


def collect_student_data_from_user() -> pd.DataFrame:
    """Collect student names, IDs, and marks interactively from the user."""
    print("\n" + "-" * 40)
    print("    INTERACTIVE MARKS ENTRY")
    print("-" * 40)

    while True:
        try:
            num_str = input("How many students would you like to enter? ").strip()
            if num_str.isdigit() and int(num_str) > 0:
                num_students = int(num_str)
                break
            print("[!] Please enter a valid positive number (e.g., 1, 2, 3...).")
        except (EOFError, KeyboardInterrupt):
            print("\nInput cancelled. Using default 1 student.")
            num_students = 1
            break

    records = []
    for i in range(1, num_students + 1):
        print(f"\n[Student {i} of {num_students}]")
        try:
            name = input("  Student Name: ").strip()
            while not name:
                name = input("  [!] Name cannot be empty. Student Name: ").strip()
        except (EOFError, KeyboardInterrupt):
            name = f"Student {i}"

        default_id = f"STU{i:03d}"
        try:
            stu_id = input(f"  Student ID [default: {default_id}]: ").strip()
            if not stu_id:
                stu_id = default_id
        except (EOFError, KeyboardInterrupt):
            stu_id = default_id

        print(f"  Enter marks for {name} (out of {MAX_MARKS_PER_SUBJECT}):")
        student_record = {"Student_ID": stu_id, "Name": name}
        for subject in SUBJECTS:
            mark = get_valid_mark(subject)
            student_record[subject] = mark

        records.append(student_record)

    df = pd.DataFrame(records)
    print(f"\n[Input Complete] Collected marks for {len(df)} student(s).")
    return df


def clean_data(df: pd.DataFrame, source_name: str = "dataset") -> pd.DataFrame:
    """Clean marks DataFrame and handle missing or invalid values.
    
    Missing values are logged and imputed using the subject's mean score
    (rounded to 1 decimal place). If all scores in a subject are missing,
    0.0 is used.
    """
    df = df.copy()
    print(f"[1/4] Processing {len(df)} student records from {source_name}...")

    # Detect missing values across subject columns
    missing_records = df[SUBJECTS].isna()
    total_missing = missing_records.sum().sum()

    if total_missing > 0:
        print(f"      Found {total_missing} missing value(s) in subject marks:")
        for subject in SUBJECTS:
            missing_students = df[df[subject].isna()]["Name"].tolist()
            if missing_students:
                valid_scores = df[subject].dropna()
                if len(valid_scores) > 0:
                    subject_mean = round(float(valid_scores.mean()), 1)
                else:
                    subject_mean = 0.0
                print(f"      - {subject}: Missing for {missing_students} -> Imputing with subject mean ({subject_mean})")
                df[subject] = df[subject].fillna(subject_mean)
    else:
        print("      No missing values detected.")

    # Ensure subject scores are float/numeric
    for subject in SUBJECTS:
        df[subject] = pd.to_numeric(df[subject], errors="coerce").fillna(0.0)

    return df


def load_and_clean_csv(file_path: Path) -> pd.DataFrame:
    """Load marks from CSV file and clean data."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input data file not found at: {file_path}")

    raw_df = pd.read_csv(file_path)
    return clean_data(raw_df, source_name=f"file '{file_path.name}'")


def choose_data_source(sample_file: Path) -> pd.DataFrame:
    """Allow user to choose between interactive entry and loading sample CSV."""
    print("=" * 70)
    print("                 STUDENT MARKS ANALYSIS SYSTEM")
    print("=" * 70)
    print("Choose data source:")
    print("  [1] Enter student marks interactively (User Input)")
    print("  [2] Load from sample CSV file (data/sample_marks.csv)")

    try:
        choice = input("\nEnter choice [1 or 2, default: 1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "2":
        return load_and_clean_csv(sample_file)
    else:
        user_df = collect_student_data_from_user()
        # Save a copy of raw user entries for record keeping
        raw_export_path = sample_file.parent / "user_entered_marks.csv"
        user_df.to_csv(raw_export_path, index=False)
        print(f"      (Raw input saved to: {raw_export_path})")
        return clean_data(user_df, source_name="user input")


def calculate_student_results(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Total Marks, Percentage, Grade, and Pass/Fail Status."""
    result_df = df.copy()

    # Compute Total Marks and Percentage using NumPy vectorization
    subject_matrix = result_df[SUBJECTS].to_numpy(dtype=float)
    total_marks = np.sum(subject_matrix, axis=1)
    max_total = len(SUBJECTS) * MAX_MARKS_PER_SUBJECT
    percentages = np.round((total_marks / max_total) * 100, 2)

    result_df["Total_Marks"] = np.round(total_marks, 1)
    result_df["Percentage"] = percentages

    # Pass / Fail criteria:
    # Must score >= PASS_PERCENTAGE overall and >= PASS_MARK_PER_SUBJECT in each subject
    subject_pass = np.all(subject_matrix >= PASS_MARK_PER_SUBJECT, axis=1)
    overall_pass = percentages >= PASS_PERCENTAGE
    is_passed = subject_pass & overall_pass
    result_df["Status"] = np.where(is_passed, "Pass", "Fail")

    # Grade determination using np.select:
    # Students who fail (overall or in any subject) receive 'F'.
    # Passing students are awarded grades based on percentage thresholds.
    grade_conditions = [
        ~is_passed,
        percentages >= 90.0,
        percentages >= 80.0,
        percentages >= 70.0,
        percentages >= 60.0,
    ]
    grade_choices = ["F", "A+", "A", "B", "C"]
    result_df["Grade"] = np.select(grade_conditions, grade_choices, default="D")

    return result_df


def compute_subject_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for each subject using NumPy functions."""
    stats_list = []
    n_students = len(df)
    ddof = 1 if n_students > 1 else 0

    for subject in SUBJECTS:
        marks = df[subject].to_numpy(dtype=float)
        stats_list.append(
            {
                "Subject": subject,
                "Mean": round(float(np.mean(marks)), 2),
                "Median": round(float(np.median(marks)), 2),
                "Std_Dev": round(float(np.std(marks, ddof=ddof)), 2),
                "Min": round(float(np.min(marks)), 1),
                "Max": round(float(np.max(marks)), 1),
                "Q1 (25th Pct)": round(float(np.percentile(marks, 25)), 2),
                "Q3 (75th Pct)": round(float(np.percentile(marks, 75)), 2),
            }
        )

    return pd.DataFrame(stats_list)


def print_console_summary(df: pd.DataFrame, subject_summary_df: pd.DataFrame) -> None:
    """Print an executive console summary of the class performance."""
    percentages = df["Percentage"].to_numpy(dtype=float)
    total_students = len(df)
    ddof = 1 if total_students > 1 else 0

    class_avg = np.mean(percentages)
    class_median = np.median(percentages)
    class_std = np.std(percentages, ddof=ddof)
    pass_count = int(np.sum(df["Status"] == "Pass"))
    pass_rate = (pass_count / total_students) * 100

    # Top and lowest scorers (supports ties)
    max_pct = np.max(percentages)
    min_pct = np.min(percentages)
    top_students = df[df["Percentage"] == max_pct][["Student_ID", "Name", "Percentage", "Grade"]].to_dict(orient="records")
    lowest_students = df[df["Percentage"] == min_pct][["Student_ID", "Name", "Percentage", "Grade"]].to_dict(orient="records")

    # Score distribution percentiles
    p10, p25, p50, p75, p90 = np.percentile(percentages, [10, 25, 50, 75, 90])

    print("\n" + "=" * 70)
    print("                    STUDENT MARKS ANALYSIS REPORT")
    print("=" * 70)

    print("\n--- CLASS OVERVIEW ---")
    print(f"Total Students Evaluated : {total_students}")
    print(f"Passed                   : {pass_count} / {total_students} ({pass_rate:.1f}%)")
    print(f"Failed                   : {total_students - pass_count} / {total_students} ({100 - pass_rate:.1f}%)")
    print(f"Class Average Percentage : {class_avg:.2f}%")
    print(f"Class Median Percentage  : {class_median:.2f}%")
    print(f"Standard Deviation       : {class_std:.2f}%")

    print("\n--- SCORE DISTRIBUTION PERCENTILES ---")
    print(f"10th Percentile : {p10:.2f}%")
    print(f"25th Percentile : {p25:.2f}% (Q1)")
    print(f"50th Percentile : {p50:.2f}% (Median)")
    print(f"75th Percentile : {p75:.2f}% (Q3)")
    print(f"90th Percentile : {p90:.2f}%")

    print("\n--- TOP SCORER(S) ---")
    for s in top_students:
        print(f" - {s['Name']} ({s['Student_ID']}): {s['Percentage']:.2f}% | Grade {s['Grade']}")

    print("\n--- LOWEST SCORER(S) ---")
    for s in lowest_students:
        print(f" - {s['Name']} ({s['Student_ID']}): {s['Percentage']:.2f}% | Grade {s['Grade']}")

    print("\n--- GRADE DISTRIBUTION ---")
    grade_counts = df["Grade"].value_counts().reindex(["A+", "A", "B", "C", "D", "F"], fill_value=0)
    for grade, count in grade_counts.items():
        bar = "█" * int(count)
        print(f"Grade {grade:<2}: {count:>2} students {bar}")

    print("\n--- SUBJECT-WISE SUMMARY ---")
    print(subject_summary_df.to_string(index=False))
    print("=" * 70 + "\n")


def export_results(
    analyzed_df: pd.DataFrame,
    subject_summary_df: pd.DataFrame,
    analyzed_path: Path,
    summary_path: Path,
) -> None:
    """Export analyzed student data and subject statistics to CSV files."""
    analyzed_df.to_csv(analyzed_path, index=False)
    subject_summary_df.to_csv(summary_path, index=False)
    print(f"[4/4] Exports saved successfully:")
    print(f"      - Analyzed marks   -> {analyzed_path}")
    print(f"      - Subject summary  -> {summary_path}")


def main() -> None:
    """Execute the end-to-end marks analysis pipeline."""
    sample_file, analyzed_csv, summary_csv = resolve_paths()

    # Step 1: Obtain student data (Interactive User Input or Sample CSV)
    raw_df = choose_data_source(sample_file)

    # Step 2: Compute student results
    print("[2/4] Calculating totals, percentages, grades, and pass/fail statuses...")
    analyzed_df = calculate_student_results(raw_df)

    # Step 3: Compute subject statistics
    print("[3/4] Computing subject-wise NumPy statistical metrics...")
    subject_summary_df = compute_subject_statistics(analyzed_df)

    # Display console summary
    print_console_summary(analyzed_df, subject_summary_df)

    # Step 4: Export artifacts
    export_results(analyzed_df, subject_summary_df, analyzed_csv, summary_csv)


if __name__ == "__main__":
    main()