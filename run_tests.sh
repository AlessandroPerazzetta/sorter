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
SAVE_RESULTS_DIR=""  # Directory to save sorting results as JSON

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
        --save-results)
            SAVE_RESULTS_DIR="$2"
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
    --save-results DIR  Save sorting results as JSON files to specified directory
                        Each file contains algorithm info, execution time, and sorted data
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
    ./run_tests.sh --save-results ./results  # Save sorting results to ./results directory
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

    # Add save-results option if specified
    if [ -n "$SAVE_RESULTS_DIR" ]; then
        # Create directory if it doesn't exist
        mkdir -p "$SAVE_RESULTS_DIR"
        cargo_cmd+=("--save-results" "$SAVE_RESULTS_DIR")
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

# Generate final HTML if requested
if [ "$GENERATE_HTML" = true ] && [ "$GENERATE_JSON_ONLY" = false ]; then
    cp html/report_template.html "$HTML_FILE"
    # Inject JSON data into the HTML
    sed "/<!-- BENCHMARK_DATA -->/{
r $JSON_FILE
d
}" "$HTML_FILE" > "${HTML_FILE}.tmp" && mv "${HTML_FILE}.tmp" "$HTML_FILE"
    echo "HTML report with charts saved to: $HTML_FILE"
fi

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed. Check $JSON_FILE for details.${NC}"
fi