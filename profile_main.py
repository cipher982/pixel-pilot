import atexit
import cProfile
import pstats
import sys

from pixelpilot.main import main


def print_profiler_stats(profiler):
    # Print the stats
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    print("\n=== cProfile Results ===\n")
    stats.print_stats(100)  # Print top 100 lines


def profile_main():
    # Create a Profile object
    profiler = cProfile.Profile()

    # Register the stats printing function to run at exit
    atexit.register(print_profiler_stats, profiler)

    # Profile the main function
    profiler.enable()
    try:
        main()
    finally:
        profiler.disable()


if __name__ == "__main__":
    # Add command line arguments to sys.argv
    sys.argv.extend(["--task-profile", "./profiles/quiz_debug.yml", "--use-firefox"])

    profile_main()
