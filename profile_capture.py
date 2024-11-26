import cProfile
import pstats

from line_profiler import LineProfiler

from pixelpilot.window_capture import WindowCapture


def main():
    capture = WindowCapture()
    window_info = capture.select_window_interactive(use_chrome=True)  # Using Chrome for consistency

    # Capture window multiple times to get better average
    for _ in range(3):
        capture.capture_window(window_info)


# cProfile approach
def profile_with_cprofile():
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats()


# line_profiler approach
def profile_with_line_profiler():
    lp = LineProfiler()
    lp_wrapper = lp(WindowCapture.capture_window)

    # Replace the method with the wrapped version
    WindowCapture.capture_window = lp_wrapper

    main()
    lp.print_stats()


if __name__ == "__main__":
    print("\n=== cProfile Results ===\n")
    profile_with_cprofile()

    print("\n=== Line-by-line Profile Results ===\n")
    profile_with_line_profiler()
