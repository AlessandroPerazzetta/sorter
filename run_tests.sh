#!/bin/bash

# Test script for Rust Sorter - runs all algorithms with various data types and sizes
#
# IMPORTANT: Algorithm timing is measured entirely within the Rust program (src/main.rs)
# using std::time::Instant. Report generation happens after each test completes and
# does not affect the timing measurements. Each cargo run is isolated.

#set -e  # Exit on any error

# Parse command line arguments
GENERATE_HTML=false
GENERATE_TEXT=false  # Text output is now optional, JSON is primary
GENERATE_JSON_ONLY=false  # New flag for execution-only mode
PARALLEL_EXECUTION=false  # New flag for parallel test execution
MAX_PARALLEL_JOBS=4  # Default number of parallel jobs
SHOW_HELP=false
PROFILE="light"  # Default profile for development
CUSTOM_SIZES=""  # Custom sizes argument
SAVE_DATA_DIR=""  # Directory to save generated data files
LOAD_DATA_DIR=""  # Directory to load data files from

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        --html|html)
            GENERATE_HTML=true
            shift
            ;;
        --text|text)
            GENERATE_TEXT=true
            shift
            ;;
        --json-only)
            GENERATE_JSON_ONLY=true
            shift
            ;;
        --parallel)
            PARALLEL_EXECUTION=true
            shift
            ;;
        --max-jobs)
            MAX_PARALLEL_JOBS="$2"
            shift
            shift
            ;;
        --profile)
            PROFILE="$2"
            shift
            shift
            ;;
        --size|--sizes)
            CUSTOM_SIZES="$2"
            shift
            shift
            ;;
        --save-data)
            SAVE_DATA_DIR="$2"
            shift
            shift
            ;;
        --load-data)
            LOAD_DATA_DIR="$2"
            shift
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$SHOW_HELP" = true ]; then
    cat << 'EOF'
Rust Sorter Benchmark Suite

USAGE:
    ./run_tests.sh [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    --profile PROFILE   Test profile: light (5 tests), medium (50 tests), full (200 tests)
                        Default: light (for development)
    --size|--sizes SIZES  Custom data sizes (comma-separated, e.g., "100,1000,10000")
                        If not specified, uses profile defaults
    --save-data DIR     Save generated data files to specified directory
                        Files will be named as 'algo_dataType_size.txt'
    --load-data DIR     Load data files from specified directory instead of generating
                        Files should be named as 'algo_dataType_size.txt'
    --json-only         Run tests and generate JSON report only (no HTML/text)
    --parallel          Run tests in parallel for faster execution
    --max-jobs NUM      Maximum number of parallel jobs (default: 4)
    --html              Generate HTML report with interactive charts
    --text              Generate text report (optional, JSON is primary)

DESCRIPTION:
    This script runs benchmarks on sorting algorithms with various data types and sizes.
    Use different profiles for different testing needs:

    - light:  3 algorithms × 2 data types × 1 size = 5 tests (fast, for development)
    - medium: 5 algorithms × 4 data types × 2 sizes = 50 tests (balanced)
    - full:   8 algorithms × 5 data types × 5 sizes = 200 tests (comprehensive)

    Results are saved to the reports/ directory with timestamped filenames.

EXAMPLES:
    ./run_tests.sh                    # Run light profile (default)
    ./run_tests.sh --profile medium   # Run medium profile
    ./run_tests.sh --profile full     # Run full comprehensive tests
    ./run_tests.sh --size "100,500,1000"  # Custom sizes with light profile
    ./run_tests.sh --profile medium --size "1000,5000"  # Medium profile with custom sizes
    ./run_tests.sh --save-data ./data  # Save generated data to ./data directory
    ./run_tests.sh --json-only        # Run tests, generate JSON only
    ./run_tests.sh --parallel --max-jobs 8 --profile full  # Run full profile in parallel
    ./run_tests.sh --html            # Generate HTML report
    ./run_tests.sh -h                # Show this help message

NOTES:
    - Tests may take several minutes to complete (especially full profile)
    - HTML reports include interactive charts for performance analysis
    - All reports are saved to the reports/ directory
EOF
    exit 0
fi

# Configure test arrays based on profile
case $PROFILE in
    "light")
        # Light profile: 3 algorithms × 2 data types × 1 size = 5 tests
        ALGORITHMS=("bubble" "selection" "insertion")
        DATA_PATTERNS=("random" "sorted")
        DATA_TYPES=("i32" "f64")
        DEFAULT_SIZES=(100)
        ;;
    "medium")
        # Medium profile: 5 algorithms × 4 data types × 2 sizes = 50 tests
        ALGORITHMS=("bubble" "selection" "insertion" "merge" "quick")
        DATA_PATTERNS=("random" "sorted" "reverse" "nearly")
        DATA_TYPES=("i32" "i64" "f32" "f64")
        DEFAULT_SIZES=(100 1000)
        ;;
    "full")
        # Full profile: 8 algorithms × 5 data types × 5 sizes = 200 tests
        ALGORITHMS=("bubble" "selection" "insertion" "merge" "quick" "heap" "parallel-bubble" "parallel-merge")
        DATA_PATTERNS=("random" "sorted" "reverse" "nearly" "duplicates")
        DATA_TYPES=("i32" "i64" "u32" "f32" "f64")
        DEFAULT_SIZES=(100 500 1000 5000 10000)
        ;;
    *)
        echo -e "${RED}Unknown profile: $PROFILE${NC}"
        echo "Available profiles: light, medium, full"
        exit 1
        ;;
esac

# Handle custom sizes if specified
if [ -n "$CUSTOM_SIZES" ]; then
    # Parse comma-separated sizes
    IFS=',' read -ra SIZES <<< "$CUSTOM_SIZES"
    # Validate that sizes are numeric
    for size in "${SIZES[@]}"; do
        if ! [[ "$size" =~ ^[0-9]+$ ]]; then
            echo -e "${RED}Error: Invalid size '$size'. Sizes must be numeric.${NC}"
            exit 1
        fi
    done
    echo "Using custom sizes: ${SIZES[*]}"
else
    SIZES=("${DEFAULT_SIZES[@]}")
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test arrays are configured based on profile above

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORTS_DIR="reports"
OUTPUT_FILE="${REPORTS_DIR}/benchmark_results_${TIMESTAMP}.txt"
HTML_FILE="${REPORTS_DIR}/benchmark_results_${TIMESTAMP}.html"

# Output files
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORTS_DIR="reports"
OUTPUT_FILE="${REPORTS_DIR}/benchmark_results_${TIMESTAMP}.txt"
HTML_FILE="${REPORTS_DIR}/benchmark_results_${TIMESTAMP}.html"
JSON_FILE="${REPORTS_DIR}/benchmark_results_${TIMESTAMP}.json"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Rust Sorter Benchmark Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Profile: $PROFILE"
if [ -n "$CUSTOM_SIZES" ]; then
    echo "Custom sizes: $CUSTOM_SIZES"
fi
echo "JSON report will be saved to: $JSON_FILE"
if [ "$GENERATE_TEXT" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    echo "Text report will be saved to: $OUTPUT_FILE"
fi
if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    echo "HTML report with charts will be saved to: $HTML_FILE"
fi
echo ""

# Function to get algorithm display name
get_algo_display_name() {
    local algo=$1
    case $algo in
        "bubble") echo "Bubble Sort" ;;
        "selection") echo "Selection Sort" ;;
        "insertion") echo "Insertion Sort" ;;
        "merge") echo "Merge Sort" ;;
        "quick") echo "Quick Sort" ;;
        "heap") echo "Heap Sort" ;;
        "parallel-bubble") echo "Parallel Bubble Sort" ;;
        "parallel-selection") echo "Parallel Selection Sort" ;;
        "parallel-merge") echo "Parallel Merge Sort" ;;
        "parallel-quick") echo "Parallel Quick Sort" ;;
        "parallel-heap") echo "Parallel Heap Sort" ;;
        "gpu-sort") echo "GPU Sort" ;;
        "gpu-bitonic") echo "GPU Bitonic Sort" ;;
        *) echo "$algo" ;;
    esac
}

# Function to get algorithm category
get_algo_category() {
    local algo=$1
    case $algo in
        "bubble"|"selection"|"insertion"|"merge"|"quick"|"heap") echo "sequential" ;;
        "parallel-"*) echo "parallel" ;;
        "gpu-"*) echo "gpu" ;;
        *) echo "unknown" ;;
    esac
}

# Function to get algorithm color
get_algo_color() {
    local algo=$1
    case $algo in
        "bubble") echo "rgba(231, 76, 60, 0.8)" ;;
        "selection") echo "rgba(243, 156, 18, 0.8)" ;;
        "insertion") echo "rgba(241, 196, 15, 0.8)" ;;
        "merge") echo "rgba(39, 174, 96, 0.8)" ;;
        "quick") echo "rgba(52, 152, 219, 0.8)" ;;
        "heap") echo "rgba(155, 89, 182, 0.8)" ;;
        "parallel-bubble") echo "rgba(192, 57, 43, 0.8)" ;;
        "parallel-selection") echo "rgba(230, 126, 34, 0.8)" ;;
        "parallel-merge") echo "rgba(34, 153, 84, 0.8)" ;;
        "parallel-quick") echo "rgba(41, 128, 185, 0.8)" ;;
        "parallel-heap") echo "rgba(142, 68, 173, 0.8)" ;;
        "gpu-sort") echo "rgba(22, 160, 133, 0.8)" ;;
        "gpu-bitonic") echo "rgba(44, 62, 80, 0.8)" ;;
        *) echo "rgba(149, 165, 166, 0.8)" ;;
    esac
}

# Declare associative arrays for HTML chart data
declare -A ALGO_TIMES
declare -A ALGO_COUNTS
declare -A DATA_PATTERN_TIMES
declare -A DATA_PATTERN_COUNTS
declare -A DATA_TYPE_TIMES
declare -A DATA_TYPE_COUNTS

# Declare associative array for JSON data (base format)
declare -A JSON_RESULTS

# Initialize counters
declare -i SUCCESSFUL_TESTS=0
declare -i FAILED_TESTS=0

# Initialize HTML results data
RESULTS_DATA=""

for algo in "${ALGORITHMS[@]}"; do
    ALGO_TIMES[$algo]=0
    ALGO_COUNTS[$algo]=0
done

for data_pattern in "${DATA_PATTERNS[@]}"; do
    DATA_PATTERN_TIMES[$data_pattern]=0
    DATA_PATTERN_COUNTS[$data_pattern]=0
done

for data_type in "${DATA_TYPES[@]}"; do
    DATA_TYPE_TIMES[$data_type]=0
    DATA_TYPE_COUNTS[$data_type]=0
done

run_test() {
    local algo=$1
    local data_pattern=$2
    local data_type=$3
    local size=$4

    # Prepare cargo run command as an array
    local cargo_cmd=("cargo" "run" "--")
    local data_filename=""
    local load_filename=""

    # Add data-type option
    cargo_cmd+=("--data-type" "$data_type")

    # Add save-data option if specified
    if [ -n "$SAVE_DATA_DIR" ]; then
        # Create directory if it doesn't exist
        mkdir -p "$SAVE_DATA_DIR"
        # Generate filename: algo_dataType_dataPattern_size.txt
        data_filename="${SAVE_DATA_DIR}/${algo}_${data_type}_${data_pattern}_${size}.txt"
        cargo_cmd+=("--save-data" "$data_filename")
    fi

    # Add load-data option if specified
    if [ -n "$LOAD_DATA_DIR" ]; then
        # Generate expected filename: algo_dataType_dataPattern_size.txt
        load_filename="${LOAD_DATA_DIR}/${algo}_${data_type}_${data_pattern}_${size}.txt"
        if [ ! -f "$load_filename" ]; then
            echo "DISPLAY|${RED}Warning: Data file not found: $load_filename${NC}" >&2
            echo "DISPLAY|${RED}Falling back to generating data${NC}" >&2
        else
            cargo_cmd+=("--load-data" "$load_filename")
        fi
    fi

    # Add the required arguments
    cargo_cmd+=("$algo" "$data_pattern" "$size")

    # Run the test and capture output
    # IMPORTANT: Timing is measured entirely within the Rust program (main.rs)
    # Report generation happens after timing is complete, so it doesn't affect measurements
    if output=$(timeout 30 "${cargo_cmd[@]}" 2>&1); then
        # Extract time from output
        time_taken=$(echo "$output" | grep "Time taken:" | sed 's/.*Time taken: \([0-9.]*\).*/\1/')
        passes=$(echo "$output" | grep "Passes:" | sed 's/.*Passes: \([0-9]*\).*/\1/' || echo "N/A")

        # Return success result with display message
        echo "DISPLAY|${YELLOW}Testing: $algo + $data_type + $data_pattern + $size${NC}"
        if [ -n "$data_filename" ]; then
            echo "DISPLAY|${BLUE}Data saved: $data_filename${NC}"
        fi
        if [ -n "$load_filename" ] && [ -f "$load_filename" ]; then
            echo "DISPLAY|${BLUE}Data loaded: $load_filename${NC}"
        fi
        echo "DISPLAY|${GREEN}✓ Success${NC} - Time: ${time_taken}s, Passes: $passes"
        echo "RESULT|SUCCESS|$algo|$data_type|$data_pattern|$size|$time_taken|$passes"
    else
        # Return failure result with display message
        echo "DISPLAY|${YELLOW}Testing: $algo + $data_type + $data_pattern + $size${NC}"
        echo "DISPLAY|${RED}✗ Failed or timed out${NC}"
        echo "RESULT|FAILED|$algo|$data_type|$data_pattern|$size"
    fi
}

process_test_result() {
    local result=$1
    IFS='|' read -r status algo data_type data_pattern size time_taken passes <<< "$result"

    if [ "$status" = "SUCCESS" ]; then
        # Save detailed results to text file (batch write for efficiency)
        if [ "$GENERATE_TEXT" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
            cat >> "$OUTPUT_FILE" << EOF
Algorithm: $algo
Data Type: $data_type
Data Pattern: $data_pattern
Size: $size
Time: ${time_taken}s
EOF
            if [ "$passes" != "N/A" ]; then
                echo "Passes: $passes" >> "$OUTPUT_FILE"
            fi
            echo "----------------------------------------" >> "$OUTPUT_FILE"
        fi

        # Store result in JSON format (base data structure)
        JSON_RESULTS["$algo-$data_type-$data_pattern-$size"]="{\"algorithm\":\"$algo\",\"data_type\":\"$data_type\",\"data_pattern\":\"$data_pattern\",\"size\":$size,\"time_seconds\":$time_taken,\"passes\":\"$passes\",\"success\":true}"

        # Add to HTML results (only if HTML generation is enabled)
        if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
            # RESULTS_DATA will be built from JSON_RESULTS at the end
            # Collect data for charts (using awk for precise floating point arithmetic)
            ALGO_TIMES[$algo]=$(awk -v current="${ALGO_TIMES[$algo]}" -v add="$time_taken" "BEGIN {print current + add}" 2>/dev/null || echo "0")
            ALGO_COUNTS[$algo]=$((ALGO_COUNTS[$algo] + 1))
            DATA_PATTERN_TIMES[$data_pattern]=$(awk -v current="${DATA_PATTERN_TIMES[$data_pattern]}" -v add="$time_taken" "BEGIN {print current + add}" 2>/dev/null || echo "0")
            DATA_PATTERN_COUNTS[$data_pattern]=$((DATA_PATTERN_COUNTS[$data_pattern] + 1))
            DATA_TYPE_TIMES[$data_type]=$(awk -v current="${DATA_TYPE_TIMES[$data_type]}" -v add="$time_taken" "BEGIN {print current + add}" 2>/dev/null || echo "0")
            DATA_TYPE_COUNTS[$data_type]=$((DATA_TYPE_COUNTS[$data_type] + 1))
        fi

        SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
    else
        # Failed test
        if [ "$GENERATE_TEXT" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
            echo "Algorithm: $algo, Data Type: $data_type, Data Pattern: $data_pattern, Size: $size - FAILED" >> "$OUTPUT_FILE"
            echo "----------------------------------------" >> "$OUTPUT_FILE"
        fi

        # Store failed result in JSON format (base data structure)
        JSON_RESULTS["$algo-$data_type-$data_pattern-$size"]="{\"algorithm\":\"$algo\",\"data_type\":\"$data_type\",\"data_pattern\":\"$data_pattern\",\"size\":$size,\"time_seconds\":null,\"passes\":null,\"success\":false}"

        # Add to HTML results
        if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
            # RESULTS_DATA will be built from JSON_RESULTS at the end
            true  # Placeholder to maintain structure
        fi

        ((FAILED_TESTS++))
    fi
    echo ""
}

# Initialize output files
if [ "$GENERATE_TEXT" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    echo "Rust Sorter Benchmark Results" > "$OUTPUT_FILE"
    echo "Generated on: $(date)" >> "$OUTPUT_FILE"
    echo "==========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    cat > "$HTML_FILE" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rust Sorter Benchmark Results</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .summary {
            background: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .summary h2 {
            margin-top: 0;
            color: #34495e;
        }
        .stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }
        .stat {
            text-align: center;
            margin: 10px;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
            color: #27ae60;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 0.9em;
        }
        .charts {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 30px;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container h3 {
            margin-top: 0;
            color: #34495e;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e8f4fd;
        }
        .success { color: #27ae60; font-weight: bold; }
        .failure { color: #e74c3c; font-weight: bold; }
        .time { font-family: 'Courier New', monospace; }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }
        .algorithm-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        .badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.2s ease;
            user-select: none;
        }
        .badge:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .badge.selected {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            border: 2px solid #fff;
        }
        .badge.unselected {
            opacity: 0.4;
        }
        .badge-bubble { background: #e74c3c; color: white; }
        .badge-selection { background: #f39c12; color: white; }
        .badge-insertion { background: #f1c40f; color: black; }
        .badge-merge { background: #27ae60; color: white; }
        .badge-quick { background: #3498db; color: white; }
        .badge-heap { background: #9b59b6; color: white; }
        .badge-parallel-bubble { background: #c0392b; color: white; }
        .badge-parallel-selection { background: #e67e22; color: white; }
        .badge-parallel-merge { background: #229954; color: white; }
        .badge-parallel-quick { background: #2980b9; color: white; }
        .badge-parallel-heap { background: #8e44ad; color: white; }
        .badge-gpu-sort { background: #16a085; color: white; }
        .badge-gpu-bitonic { background: #2c3e50; color: white; }
        .filter-controls {
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
        .filter-controls h4 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 0.9em;
        }
        .filter-buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 5px 12px;
            border: none;
            border-radius: 4px;
            background: #6c757d;
            color: white;
            cursor: pointer;
            font-size: 0.8em;
            transition: background-color 0.2s;
        }
        .filter-btn:hover {
            background: #5a6268;
        }
        .filter-btn.select-all {
            background: #28a745;
        }
        .filter-btn.select-all:hover {
            background: #218838;
        }
        .filter-btn.clear-all {
            background: #dc3545;
        }
        .filter-btn.clear-all:hover {
            background: #c82333;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🦀 Rust Sorter Benchmark Results</h1>
        <div class="summary">
            <h2>📊 Test Summary</h2>
            <div class="stats" id="summary-stats">
                <!-- Stats will be inserted here -->
            </div>
        </div>

        <h2>📈 Performance Charts</h2>
        <div class="charts">
            <div class="chart-container">
                <h3>Average Time by Algorithm</h3>
                <canvas id="algorithmChart" width="400" height="300"></canvas>
            </div>
            <div class="chart-container">
                <h3>Average Time by Data Type</h3>
                <canvas id="dataTypeChart" width="400" height="300"></canvas>
            </div>
        </div>

        <h2>🏷️ Algorithms Tested</h2>
        <div class="algorithm-badges">
            <!-- Algorithm badges will be inserted here -->
        </div>

        <h2>📋 Detailed Results</h2>
        <div class="filter-controls">
            <h4>Filter Results by Algorithm:</h4>
            <div class="filter-buttons">
                <button class="filter-btn select-all" onclick="selectAllAlgorithms()">Select All</button>
                <button class="filter-btn clear-all" onclick="clearAllAlgorithms()">Clear All</button>
            </div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Algorithm</th>
                    <th>Data Type</th>
                    <th>Size</th>
                    <th>Time (s)</th>
                    <th>Passes</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody id="results-body">
                <!-- Results will be inserted here -->
            </tbody>
        </table>

        <div class="footer">
            <p>Generated on: $(date) | Total Tests: <span id="total-tests"></span></p>
        </div>
    </div>

    <script>
        // Chart data will be inserted here
        <!-- CHART_DATA -->

        // Algorithm filtering functionality
        let selectedAlgorithms = new Set();

        function initializeAlgorithmFilter() {
            // Get all algorithm badges
            const badges = document.querySelectorAll('.badge[data-algorithm]');
            const tableRows = document.querySelectorAll('#results-body tr');

            // Initially select all algorithms
            badges.forEach(badge => {
                const algorithm = badge.getAttribute('data-algorithm');
                selectedAlgorithms.add(algorithm);
                badge.classList.add('selected');
            });

            // Add click event listeners to badges
            badges.forEach(badge => {
                badge.addEventListener('click', function() {
                    const algorithm = this.getAttribute('data-algorithm');
                    toggleAlgorithm(algorithm);
                });
            });

            // Initial filter application
            applyFilter();
        }

        function toggleAlgorithm(algorithm) {
            const badge = document.querySelector(`.badge[data-algorithm="${algorithm}"]`);

            if (selectedAlgorithms.has(algorithm)) {
                selectedAlgorithms.delete(algorithm);
                badge.classList.remove('selected');
                badge.classList.add('unselected');
            } else {
                selectedAlgorithms.add(algorithm);
                badge.classList.add('selected');
                badge.classList.remove('unselected');
            }

            applyFilter();
        }

        function selectAllAlgorithms() {
            const badges = document.querySelectorAll('.badge[data-algorithm]');
            badges.forEach(badge => {
                const algorithm = badge.getAttribute('data-algorithm');
                selectedAlgorithms.add(algorithm);
                badge.classList.add('selected');
                badge.classList.remove('unselected');
            });
            applyFilter();
        }

        function clearAllAlgorithms() {
            const badges = document.querySelectorAll('.badge[data-algorithm]');
            badges.forEach(badge => {
                const algorithm = badge.getAttribute('data-algorithm');
                selectedAlgorithms.delete(algorithm);
                badge.classList.remove('selected');
                badge.classList.add('unselected');
            });
            applyFilter();
        }

        function applyFilter() {
            const tableRows = document.querySelectorAll('#results-body tr');
            let visibleCount = 0;

            tableRows.forEach(row => {
                const algorithm = row.getAttribute('data-algorithm');
                if (selectedAlgorithms.has(algorithm)) {
                    row.style.display = '';
                    visibleCount++;
                } else {
                    row.style.display = 'none';
                }
            });

            // Update summary stats to show filtered results
            updateFilteredStats(visibleCount);
        }

        function updateFilteredStats(visibleCount) {
            const totalRows = document.querySelectorAll('#results-body tr').length;
            const summaryContainer = document.getElementById('summary-stats');

            // Calculate filtered successful/failed counts
            let filteredSuccessful = 0;
            let filteredFailed = 0;

            document.querySelectorAll('#results-body tr').forEach(row => {
                if (row.style.display !== 'none') {
                    const statusCell = row.querySelector('td:last-child');
                    if (statusCell && statusCell.textContent.includes('Success')) {
                        filteredSuccessful++;
                    } else if (statusCell && statusCell.textContent.includes('Failed')) {
                        filteredFailed++;
                    }
                }
            });

            const filteredStats = `
                <div class="stat">
                    <div class="stat-number">${visibleCount}</div>
                    <div class="stat-label">Filtered Tests</div>
                </div>
                <div class="stat">
                    <div class="stat-number" style="color: #27ae60;">${filteredSuccessful}</div>
                    <div class="stat-label">Successful</div>
                </div>
                <div class="stat">
                    <div class="stat-number" style="color: #e74c3c;">${filteredFailed}</div>
                    <div class="stat-label">Failed</div>
                </div>
            `;

            summaryContainer.innerHTML = filteredStats;
        }

        // Initialize filtering when page loads
        document.addEventListener('DOMContentLoaded', function() {
            initializeAlgorithmFilter();
        });
    </script>
</body>
</html>
EOF
fi

# Build the project first
echo -e "${BLUE}Building project...${NC}"
if ! cargo build --release; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
echo -e "${GREEN}Build successful${NC}"
echo ""

# Run all combinations
total_tests=$(( ${#ALGORITHMS[@]} * ${#DATA_PATTERNS[@]} * ${#DATA_TYPES[@]} * ${#SIZES[@]} ))
current_test=1

if [ "$PARALLEL_EXECUTION" = true ]; then
    echo -e "${YELLOW}Running tests in parallel (max $MAX_PARALLEL_JOBS jobs)${NC}"

    # Create a temporary directory for parallel job outputs
    TEMP_DIR=$(mktemp -d)
    pids=()

    for algo in "${ALGORITHMS[@]}"; do
        for data_pattern in "${DATA_PATTERNS[@]}"; do
            for data_type in "${DATA_TYPES[@]}"; do
                for size in "${SIZES[@]}"; do
                    echo -e "${BLUE}Test $current_test/$total_tests${NC}"

                    # Run test in background and capture result
                    (
                        run_test "$algo" "$data_pattern" "$data_type" "$size" > "$TEMP_DIR/result_${current_test}.txt"
                    ) &

                    pids+=($!)
                    ((current_test++))

                    # Limit concurrent jobs
                    if [ ${#pids[@]} -ge $MAX_PARALLEL_JOBS ]; then
                        # Wait for first job to complete
                        wait "${pids[0]}"
                        pids=("${pids[@]:1}")
                    fi
                done
            done
        done
    done

    # Wait for remaining jobs
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    echo -e "${GREEN}All parallel tests completed${NC}"

    # Process results sequentially
    for ((i=1; i<=total_tests; i++)); do
        result_file="$TEMP_DIR/result_${i}.txt"
        if [ -f "$result_file" ]; then
            while IFS= read -r line; do
                if [[ "$line" =~ ^DISPLAY\| ]]; then
                    # Display the message
                    message="${line#DISPLAY|}"
                    echo -e "$message"
                elif [[ "$line" =~ ^RESULT\| ]]; then
                    # Process the result
                    result="${line#RESULT|}"
                    process_test_result "$result"
                fi
            done < "$result_file"
        fi
    done

    # Clean up temp directory
    rm -rf "$TEMP_DIR"
else
    # Sequential execution (original logic)
    for algo in "${ALGORITHMS[@]}"; do
        for data_pattern in "${DATA_PATTERNS[@]}"; do
            for data_type in "${DATA_TYPES[@]}"; do
                for size in "${SIZES[@]}"; do
                    echo -e "${BLUE}Test $current_test/$total_tests${NC}"
                    # Run test and process output lines
                    while IFS= read -r line; do
                        if [[ "$line" =~ ^DISPLAY\| ]]; then
                            # Display the message
                            message="${line#DISPLAY|}"
                            echo -e "$message"
                        elif [[ "$line" =~ ^RESULT\| ]]; then
                            # Process the result
                            result="${line#RESULT|}"
                            process_test_result "$result"
                        fi
                    done < <(run_test "$algo" "$data_pattern" "$data_type" "$size")
                    ((current_test++))
                done
            done
        done
    done
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
if [ "$GENERATE_TEXT" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    echo "Text results saved to: $OUTPUT_FILE"
fi
echo "JSON results saved to: $JSON_FILE"
if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    echo "HTML report with charts saved to: $HTML_FILE"
fi
echo ""

# Show summary
echo -e "${BLUE}Summary:${NC}"
echo "Total tests run: $total_tests"
echo "Successful: $SUCCESSFUL_TESTS"
echo "Failed: $FAILED_TESTS"

# Generate final HTML if requested
if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    # Generate chart data from JSON_RESULTS
    ALGO_LABELS=""
    ALGO_AVG_TIMES=""
    ALGO_COLORS=""
    
    # Calculate algorithm aggregates from JSON_RESULTS
    declare -A algo_times
    declare -A algo_counts
    declare -A algo_data_type_times
    declare -A algo_data_type_counts
    for key in "${!JSON_RESULTS[@]}"; do
        json_data="${JSON_RESULTS[$key]}"
        # Extract algorithm, data_type, time, and success from JSON
        algo=$(echo "$json_data" | sed 's/.*"algorithm":"\([^"]*\)".*/\1/')
        data_type=$(echo "$json_data" | sed 's/.*"data_type":"\([^"]*\)".*/\1/')
        time_val=$(echo "$json_data" | sed 's/.*"time_seconds":\([^,}]*\).*/\1/')
        success=$(echo "$json_data" | sed 's/.*"success":\([^}]*\).*/\1/')
        
        if [ "$success" = "true" ] && [ "$time_val" != "null" ]; then
            algo_times[$algo]=$(awk -v current="${algo_times[$algo]}" -v add="$time_val" "BEGIN {print current + add}" 2>/dev/null || echo "0")
            algo_counts[$algo]=$((algo_counts[$algo] + 1))
            algo_data_type_times["$algo-$data_type"]=$(awk -v current="${algo_data_type_times[$algo-$data_type]}" -v add="$time_val" "BEGIN {print current + add}" 2>/dev/null || echo "0")
            algo_data_type_counts["$algo-$data_type"]=$((algo_data_type_counts[$algo-$data_type] + 1))
        fi
    done
    
    for algo in "${ALGORITHMS[@]}"; do
        if [ "${algo_counts[$algo]}" -gt 0 ]; then
            avg_time=$(awk "BEGIN {printf \"%.6f\", ${algo_times[$algo]} / ${algo_counts[$algo]}}")
            display_name=$(get_algo_display_name "$algo")
            color=$(get_algo_color "$algo")
            ALGO_LABELS="${ALGO_LABELS}\"${display_name}\", "
            ALGO_AVG_TIMES="${ALGO_AVG_TIMES}${avg_time}, "
            ALGO_COLORS="${ALGO_COLORS}\"${color}\", "
        fi
    done
    ALGO_LABELS="${ALGO_LABELS%, }"
    ALGO_AVG_TIMES="${ALGO_AVG_TIMES%, }"
    ALGO_COLORS="${ALGO_COLORS%, }"

    DATA_TYPE_LABELS=""
    DATA_TYPE_DATASETS=""
    
    # Get all data types that were tested
    declare -A tested_data_types
    for key in "${!JSON_RESULTS[@]}"; do
        json_data="${JSON_RESULTS[$key]}"
        data_type=$(echo "$json_data" | sed 's/.*"data_type":"\([^"]*\)".*/\1/')
        tested_data_types[$data_type]=1
    done

    # Sort data types for consistent ordering
    for data_type in $(printf '%s\n' "${!tested_data_types[@]}" | sort); do
        DATA_TYPE_LABELS="${DATA_TYPE_LABELS}\"${data_type}\", "
    done
    DATA_TYPE_LABELS="${DATA_TYPE_LABELS%, }"

    # Generate datasets for each algorithm
    for algo in "${ALGORITHMS[@]}"; do
        display_name=$(get_algo_display_name "$algo")
        color=$(get_algo_color "$algo")
        algo_data=""
        
        for data_type in $(printf '%s\n' "${!tested_data_types[@]}" | sort); do
            key="$algo-$data_type"
            if [ "${algo_data_type_counts[$key]}" -gt 0 ]; then
                avg_time=$(awk "BEGIN {printf \"%.6f\", ${algo_data_type_times[$key]} / ${algo_data_type_counts[$key]}}")
                algo_data="${algo_data}${avg_time}, "
            else
                algo_data="${algo_data}null, "
            fi
        done
        algo_data="${algo_data%, }"
        
        DATA_TYPE_DATASETS="${DATA_TYPE_DATASETS}{
            label: '${display_name}',
            data: [${algo_data}],
            backgroundColor: '${color}',
            borderColor: '${color}',
            borderWidth: 2,
            fill: false,
            tension: 0.4
        }, "
    done
    DATA_TYPE_DATASETS="${DATA_TYPE_DATASETS%, }"

    CHART_SCRIPT="
        // Algorithm performance chart
        const algoCtx = document.getElementById('algorithmChart').getContext('2d');
        new Chart(algoCtx, {
            type: 'bar',
            data: {
                labels: [${ALGO_LABELS}],
                datasets: [{
                    label: 'Average Time (seconds)',
                    data: [${ALGO_AVG_TIMES}],
                    backgroundColor: [${ALGO_COLORS}],
                    borderColor: [${ALGO_COLORS}],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Algorithm'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Average Execution Time by Sorting Algorithm'
                    }
                }
            }
        });

        // Data type performance chart
        const dataCtx = document.getElementById('dataTypeChart').getContext('2d');
        new Chart(dataCtx, {
            type: 'line',
            data: {
                labels: [${DATA_TYPE_LABELS}],
                datasets: [${DATA_TYPE_DATASETS}]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Time (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Data Type'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Average Execution Time by Data Type and Algorithm'
                    }
                }
            }
        });
"

    # Insert summary stats
    sed -i "s|<!-- Stats will be inserted here -->|<div class=\"stat\"><div class=\"stat-number\">$total_tests</div><div class=\"stat-label\">Total Tests</div></div><div class=\"stat\"><div class=\"stat-number\" style=\"color: #27ae60;\">$SUCCESSFUL_TESTS</div><div class=\"stat-label\">Successful</div></div><div class=\"stat\"><div class=\"stat-number\" style=\"color: #e74c3c;\">$FAILED_TESTS</div><div class=\"stat-label\">Failed</div></div>|" "$HTML_FILE"

    # Generate results table from JSON_RESULTS
    RESULTS_DATA=""
    for key in "${!JSON_RESULTS[@]}"; do
        json_data="${JSON_RESULTS[$key]}"
        # Extract fields from JSON
        algo=$(echo "$json_data" | sed 's/.*"algorithm":"\([^"]*\)".*/\1/')
        data_type=$(echo "$json_data" | sed 's/.*"data_type":"\([^"]*\)".*/\1/')
        size=$(echo "$json_data" | sed 's/.*"size":\([^,}]*\).*/\1/')
        time_val=$(echo "$json_data" | sed 's/.*"time_seconds":\([^,}]*\).*/\1/')
        passes=$(echo "$json_data" | sed 's/.*"passes":\([^,}]*\).*/\1/')
        success=$(echo "$json_data" | sed 's/.*"success":\([^}]*\).*/\1/')
        
        badge_class="badge-${algo}"
        display_name=$(get_algo_display_name "$algo")
        
        if [ "$success" = "true" ]; then
            status_class="success"
            status_text="✓ Success"
            display_time="${time_val}"
            display_passes="$passes"
        else
            status_class="failure"
            status_text="✗ Failed"
            display_time="N/A"
            display_passes="N/A"
        fi
        
        RESULTS_DATA="${RESULTS_DATA}<tr data-algorithm=\"${algo}\"><td><span class=\"badge ${badge_class}\">${display_name}</span></td><td>${data_type}</td><td>${size}</td><td class=\"time\">${display_time}</td><td>${display_passes}</td><td class=\"${status_class}\">${status_text}</td></tr>"
    done

    # Generate dynamic algorithm badges based on tested algorithms
    declare -A tested_algorithms
    for key in "${!JSON_RESULTS[@]}"; do
        json_data="${JSON_RESULTS[$key]}"
        algo=$(echo "$json_data" | sed 's/.*"algorithm":"\([^"]*\)".*/\1/')
        tested_algorithms[$algo]=1
    done

    # Generate badges HTML
    BADGES_HTML=""
    
    # Group algorithms by category
    SEQUENTIAL_BADGES=""
    PARALLEL_BADGES=""
    GPU_BADGES=""
    
    for algo in "${!tested_algorithms[@]}"; do
        display_name=$(get_algo_display_name "$algo")
        badge_class="badge-${algo}"
        
        case $algo in
            "bubble"|"selection"|"insertion"|"merge"|"quick"|"heap")
                SEQUENTIAL_BADGES="${SEQUENTIAL_BADGES}<span class=\"badge ${badge_class}\" data-algorithm=\"${algo}\">${display_name}</span> "
                ;;
            "parallel-"*)
                PARALLEL_BADGES="${PARALLEL_BADGES}<span class=\"badge ${badge_class}\" data-algorithm=\"${algo}\">${display_name}</span> "
                ;;
            "gpu-"*)
                GPU_BADGES="${GPU_BADGES}<span class=\"badge ${badge_class}\" data-algorithm=\"${algo}\">${display_name}</span> "
                ;;
        esac
    done

    if [ -n "$SEQUENTIAL_BADGES" ]; then
        BADGES_HTML="${BADGES_HTML}<h3>Sequential Algorithms</h3>${SEQUENTIAL_BADGES}"
    fi
    if [ -n "$PARALLEL_BADGES" ]; then
        BADGES_HTML="${BADGES_HTML}<h3>Parallel CPU Algorithms</h3>${PARALLEL_BADGES}"
    fi
    if [ -n "$GPU_BADGES" ]; then
        BADGES_HTML="${BADGES_HTML}<h3>GPU-Accelerated Algorithms</h3>${GPU_BADGES}"
    fi

    # Insert badges into HTML
    sed -i "s|<!-- Algorithm badges will be inserted here -->|${BADGES_HTML}|" "$HTML_FILE"

    # Insert results data - use a safer approach for large datasets
    echo "$RESULTS_DATA" > /tmp/results_data.html
    sed -i '/<!-- Results will be inserted here -->/r /tmp/results_data.html' "$HTML_FILE"
    sed -i '/<!-- Results will be inserted here -->/d' "$HTML_FILE"

    # Insert chart script - use a safer approach
    echo "$CHART_SCRIPT" > /tmp/chart_data.js
    sed -i '/<!-- CHART_DATA -->/r /tmp/chart_data.js' "$HTML_FILE"
    sed -i '/<!-- CHART_DATA -->/d' "$HTML_FILE"

    # Insert total tests in footer
    sed -i "s|<span id=\"total-tests\"></span>|$total_tests|" "$HTML_FILE"
fi

# Generate JSON report from JSON_RESULTS array
echo "Generating JSON report..."
{
    echo "{"
    echo "  \"metadata\": {"
    echo "    \"generated_at\": \"$(date -Iseconds)\","
    echo "    \"total_tests\": $total_tests,"
    echo "    \"successful_tests\": $SUCCESSFUL_TESTS,"
    echo "    \"failed_tests\": $FAILED_TESTS"
    echo "  },"
    echo "  \"results\": ["
    
    # Output all results from JSON_RESULTS array
    first=true
    for key in "${!JSON_RESULTS[@]}"; do
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        echo "    ${JSON_RESULTS[$key]}"
    done
    
    echo ""
    echo "  ]"
    echo "}"
} > "$JSON_FILE"

echo "JSON report saved to: $JSON_FILE"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check $JSON_FILE for details.${NC}"
fi