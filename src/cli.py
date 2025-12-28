import argparse
import traceback

from runner import AnalysisRunner


def main():
    parser = argparse.ArgumentParser(description="Administrative Procedure Cost Analysis CLI")
    parser.add_argument(
        "city_id",
        nargs="?",
        default=None,
        help="Target Municipality ID (e.g., 13105). Required if --all-city is not specified.",
    )
    parser.add_argument("--procedure", default="児童手当 認定請求", help="Target Procedure Name (default: 児童手当 認定請求)")
    parser.add_argument("--output-dir", default="output/graphs", help="Directory to save results")
    parser.add_argument("--all-city", action="store_true", help="Run analysis for all target cities.")

    args = parser.parse_args()

    # AnalysisRunnerを初期化
    runner = AnalysisRunner(procedure_name=args.procedure, output_dir=args.output_dir)

    if args.all_city:
        print("Running analysis for all target cities...")
        results = runner.run_for_all_targets()

        failed_cities = [city_id for city_id, result_path in results.items() if result_path is None]

        print("-" * 50)
        if failed_cities:
            print(f"Analysis completed with failures for the following cities: {', '.join(failed_cities)}")
        else:
            print("All analyses completed successfully.")
    elif args.city_id:
        try:
            runner.run_single(args.city_id)
        except Exception:
            print(f"An error occurred during analysis for city {args.city_id}:")
            traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
